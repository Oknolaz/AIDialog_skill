from vasisualy.skills.vas_skill.vas_skill import Skill  # Импорт родительского класса навыков
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Grossmend/rudialogpt3_medium_based_on_gpt2")
model = AutoModelForCausalLM.from_pretrained("Grossmend/rudialogpt3_medium_based_on_gpt2")
step = 0
chat_history_ids = None


class AIDialog(Skill):
    def get_length_param(self, text: str) -> str:
                tokens_count = len(tokenizer.encode(text))
                if tokens_count <= 15:
                    len_param = '1'
                elif tokens_count <= 50:
                    len_param = '2'
                elif tokens_count <= 256:
                    len_param = '3'
                else:
                    len_param = '-'
                return len_param


    def first_run(self, user_message):
        if super(AIDialog, self)._is_triggered(user_message, super(AIDialog, self)._get_triggers()):
            user_input = tokenizer.encode(f"|0|{self.get_length_param(user_message)}|" + user_message + tokenizer.eos_token +  "|1|1|", return_tensors="pt")

            chat_history_ids = model.generate(
                user_input,
                num_return_sequences=1,
                max_length=256,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=100,
                top_p=0.9,
                temperature = 0.7,
                mask_token_id=tokenizer.mask_token_id,
                eos_token_id=tokenizer.eos_token_id,
                unk_token_id=tokenizer.unk_token_id,
                pad_token_id=tokenizer.pad_token_id,
                device='cpu',
            )

            toSpeak = tokenizer.decode(chat_history_ids[:, user_input.shape[-1]:][0], skip_special_tokens=True)

            super(AIDialog, self).run_loop()

            return toSpeak
        else:
            return ''


    def main(self, user_message):

        if not user_message:
            return ''

        if super(AIDialog, self)._is_triggered_to_exit(user_message, super(AIDialog, self)._get_exit_triggers()):
            toSpeak = "Завершение работы навыка..."

            super(AIDialog, self).exit_loop()  # Завершение цикла.

            return toSpeak

        else:
            global step, chat_history_ids

            user_input = tokenizer.encode(f"|0|{self.get_length_param(user_message)}|" + user_message + tokenizer.eos_token +  "|1|1|", return_tensors="pt")

            # добавление input токенов нового пользователя в историю чата
            bot_input_ids = torch.cat([chat_history_ids, user_input], dim=-1) if step > 0 else user_input

            step += 1

            # generated a response
            chat_history_ids = model.generate(
                bot_input_ids,
                num_return_sequences=1,
                max_length=1024,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature = 0.6,
                mask_token_id=tokenizer.mask_token_id,
                eos_token_id=tokenizer.eos_token_id,
                unk_token_id=tokenizer.unk_token_id,
                pad_token_id=tokenizer.pad_token_id,
                device='cpu',
            )

            toSpeak = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
            return toSpeak


def main(user_message):
    skill = AIDialog("AIDialog", user_message, loop=True)  # Вывод сообщения, переданного навыком, пользователю.
    return skill.first_run(user_message)

def loop(user_message):
    skill = AIDialog("AIDialog", user_message, loop=True)
    return skill.main(user_message)
