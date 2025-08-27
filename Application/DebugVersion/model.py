import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BLIP2VLM:
    
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"BLIP2VLM 初始化 - 使用設備: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU 名稱: {torch.cuda.get_device_name()}")
            print(f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
        else:
            print("⚠️ 警告: 沒有可用的 CUDA GPU，將使用 CPU (會很慢)")
        
        MODEL_ID = "Salesforce/blip2-opt-2.7b"
        self.processor = Blip2Processor.from_pretrained(MODEL_ID, force_download=True)
        
        self._load_model(MODEL_ID)
    
    def _load_model(self, model_id: str):
        try:
            print("嘗試使用 device_map='auto' 載入模型...")
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_id, 
                device_map="auto", 
                torch_dtype=torch.float16
            )
            print("✅ 成功使用 device_map='auto' 載入模型")
        except Exception as e:
            print(f"❌ 使用 device_map 失敗，改用傳統方式: {e}")
            print(f"使用傳統方式載入模型到 {self.device}...")
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16
            ).to(self.device)
            print(f"✅ 成功載入模型到 {self.device}")

    def score_image_with_texts(self, pil_image: Image.Image, texts: list) -> dict:
        try:
            question = "Look at this image carefully. Is there a vehicle parked on or over a red line? Red lines mean no parking. Answer only Yes or No."
            inputs = self.processor(images=pil_image, text=question, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=3,
                    do_sample=False
                )
            
            in_len = inputs["input_ids"].shape[1]
            answer = self.processor.decode(outputs[0, in_len:], skip_special_tokens=True).strip()
            
            first_word = answer.split()[0].strip(",. ").capitalize() if answer else ""
            if first_word not in {"Yes", "No"}:
                full_answer = self.processor.decode(outputs[0], skip_special_tokens=True)
                first_word = "Yes" if "yes" in full_answer.lower() else ("No" if "no" in full_answer.lower() else "No")
            
            if first_word == "Yes":
                illegal_score = 0.8
                legal_score = 0.2
            else:
                illegal_score = 0.2
                legal_score = 0.8
            
            return {
                "The vehicle is illegally parked on a red line.": illegal_score,
                "The vehicle is parked legally.": legal_score
            }
        except Exception as e:
            logger.error(f"BLIP-2 分析錯誤: {e}")
            return {
                "The vehicle is illegally parked on a red line.": 0.5,
                "The vehicle is parked legally.": 0.5
            }

    def detailed_analysis(self, pil_image: Image.Image) -> dict:
        try:
            q1 = "Look at this image carefully. Is there a vehicle parked on or over a red line? Red lines indicate no-parking zones. Answer only Yes or No."
            inp1 = self.processor(images=pil_image, text=q1, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out1 = self.model.generate(
                    **inp1,
                    max_new_tokens=5,
                    do_sample=False,
                    num_beams=1
                )
            
            in_len1 = inp1["input_ids"].shape[1]
            ans1 = self.processor.decode(out1[0, in_len1:], skip_special_tokens=True).strip()
            
            first = ans1.split()[0].strip(",. ").capitalize() if ans1 else ""
            if first not in {"Yes", "No"}:
                full1 = self.processor.decode(out1[0], skip_special_tokens=True)
                full_lower = full1.lower()
                if "yes" in full_lower and "no" not in full_lower:
                    first = "Yes"
                elif "no" in full_lower and "yes" not in full_lower:
                    first = "No"
                else:
                    first = "No"
            
            illegal = first
            
            if illegal == "Yes":
                q2 = "Look at the red line in the image. Why is this vehicle parked illegally? Is it touching or crossing the red line? Give a specific reason."
            else:
                q2 = "Look at the parking area in the image. Why is this vehicle parked legally? Is it in a designated parking spot? Give a specific reason."
            
            inp2 = self.processor(images=pil_image, text=q2, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                out2 = self.model.generate(
                    **inp2,
                    max_new_tokens=30,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.8,
                    num_beams=2
                )
            
            dec2_full = self.processor.decode(out2[0], skip_special_tokens=True)
            
            if ":" in dec2_full:
                reason = dec2_full.split(":", 1)[-1].strip()
            else:
                reason = dec2_full.strip()
            
            junk = [
                "because", "the", "this", "vehicle", "is", "parked", "illegally", "legally",
                "reason", "answer", "format", "brief", "short", "evidence"
            ]
            words = reason.split()
            filtered_words = [word for word in words if word.lower() not in junk]
            reason = " ".join(filtered_words).strip()
            
            if not reason or len(reason) < 3:
                if illegal == "Yes":
                    reason = "Vehicle appears to be parked in a restricted area"
                else:
                    reason = "Vehicle appears to be parked in a designated area"
            
            return {
                "is_illegal": illegal == "Yes",
                "confidence": illegal,
                "reason": reason,
                "full_response": f"Illegal: {illegal}\nReason: {reason}"
            }
        except Exception as e:
            logger.error(f"BLIP-2 詳細分析錯誤: {e}")
            return {
                "is_illegal": False,
                "confidence": "Unknown",
                "reason": "Analysis failed",
                "full_response": "Analysis failed due to error"
            }

    def image_captioning(self, pil_image: Image.Image) -> str:
        try:
            inputs = self.processor(pil_image, return_tensors="pt").to(self.device, torch.float16)
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs)
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True).strip()
            return caption
        except Exception as e:
            logger.error(f"Image Captioning 錯誤: {e}")
            return "Caption generation failed"

    def vqa_question(self, pil_image: Image.Image, question: str) -> str:
        try:
            formatted_question = f"Question: {question}\nAnswer:"
            
            inputs = self.processor(images=pil_image, text=formatted_question, return_tensors="pt").to(self.device, torch.float16)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=20, 
                    num_beams=3, 
                    do_sample=False, 
                    temperature=0.1,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
            
            full_response = self.processor.decode(generated_ids[0], skip_special_tokens=True).strip()
            
            if "Answer:" in full_response:
                answer = full_response.split("Answer:", 1)[-1].strip()
            else:
                answer = full_response
            
            answer = answer.replace(question, "").strip()
            
            if not answer or answer == question or len(answer) < 2:
                if "how many" in question.lower():
                    answer = "Unable to determine exact count"
                elif "what" in question.lower():
                    answer = "Unable to identify specific details"
                else:
                    answer = "Unable to provide a clear answer"
            
            return answer
        except Exception as e:
            logger.error(f"VQA 錯誤: {e}")
            return "VQA failed"

    def get_device_info(self) -> dict:
        info = {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "model_loaded": hasattr(self, 'model')
        }
        
        if torch.cuda.is_available():
            info.update({
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "gpu_memory_allocated": torch.cuda.memory_allocated() / (1024**3),
                "gpu_memory_reserved": torch.cuda.memory_reserved() / (1024**3)
            })
        
        return info

    def cleanup(self):
        try:
            if hasattr(self, 'model'):
                del self.model
            if hasattr(self, 'processor'):
                del self.processor
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            logger.info("BLIP2VLM 資源清理完成")
        except Exception as e:
            logger.error(f"BLIP2VLM 資源清理失敗: {e}")
