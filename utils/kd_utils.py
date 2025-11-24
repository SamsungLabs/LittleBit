import torch
import torch.nn.functional as F
from transformers import Trainer


class KDTrainer(Trainer):
    def __init__(self, teacher_model, l2l_loss_scale, *args, **kwargs):
        super(KDTrainer, self).__init__(*args, **kwargs)

        self.teacher_model = teacher_model
        # self.teacher_model = self.teacher_model.eval()
        self.l2l_loss_scale = l2l_loss_scale

    def ce_loss(self, student_logits, teacher_logits):
        model_output_log_prob = F.log_softmax(student_logits, dim=-1)
        real_output_soft = F.softmax(teacher_logits, dim=-1)

        return F.kl_div(model_output_log_prob, real_output_soft, reduction="batchmean")

    def mse_loss(self, student_logits, teacher_logits):
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            return F.mse_loss(student_logits, teacher_logits)

    # Implement KD functions
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs["output_hidden_states"] = True

        # Teacher Model Inference (KD)
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)

        teacher_logits = teacher_outputs.get("logits")
        teacher_reps = teacher_outputs.hidden_states[1:]
        del teacher_outputs

        # Student Model Inference
        outputs = model(**inputs)

        student_logits = outputs.get("logits")
        student_reps = outputs.hidden_states[1:]

        if not return_outputs:
            del outputs

        kd_loss = self.ce_loss(student_logits, teacher_logits)

        l2l_loss = 0
        for student_rep, teacher_rep in zip(student_reps, teacher_reps):
            tmp_loss = self.mse_loss(student_rep, teacher_rep)
            l2l_loss += tmp_loss
        l2l_loss = self.l2l_loss_scale * l2l_loss

        loss = kd_loss + l2l_loss

        self.log({
            "l2l_loss": l2l_loss.item(),
            "kd_loss": kd_loss.item(),
        })

        return (loss, outputs) if return_outputs else loss
