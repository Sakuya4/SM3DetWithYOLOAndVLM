import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration


"""loading pre-train processor and model"""
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
"""loading pre-train processor and model done"""

"""image process"""
imgPath = "./image.png"
rawImg = Image.open(imgPath).convert("RGB")
"""image process done"""

"""model visual question answering"""
inputs_captioning = processor(rawImg, return_tensors="pt").to(device, torch.float16)
generated_ids_captioning = model.generate(**inputs_captioning)
generated_text_captioning = processor.batch_decode(generated_ids_captioning, skip_special_tokens=True)[0].strip()
print("Image Captioning:", generated_text_captioning)

# Q1
question = "Question: How many cars are in the first sub-image? Answer:"
inputs_vqa = processor(rawImg, text=question, return_tensors="pt").to(device, torch.float16)
generated_ids_vqa = model.generate(**inputs_vqa, max_new_tokens=15, num_beams=3, do_sample=False, temperature=0.1)
generated_text_vqa = processor.batch_decode(generated_ids_vqa, skip_special_tokens=True)[0].strip()
print("VQA (first sub-image):", generated_text_vqa)
# Q2
question2 = "Question: How many cars are in the second sub-image? Answer:"
inputs_vqa2 = processor(rawImg, text=question2, return_tensors="pt").to(device, torch.float16)
generated_ids_vqa2 = model.generate(**inputs_vqa2, max_new_tokens=15, num_beams=3, do_sample=False, temperature=0.1)
generated_text_vqa2 = processor.batch_decode(generated_ids_vqa2, skip_special_tokens=True)[0].strip()
print("VQA (second sub-image):", generated_text_vqa2)

#Q3-Simple question
question3 = "What do you see in the first image?"
inputs_vqa3 = processor(rawImg, text=question3, return_tensors="pt").to(device, torch.float16)
generated_ids_vqa3 = model.generate(**inputs_vqa3, max_new_tokens=20, num_beams=3, do_sample=False, temperature=0.1)
generated_text_vqa3 = processor.batch_decode(generated_ids_vqa3, skip_special_tokens=True)[0].strip()
print("VQA (description of first image):", generated_text_vqa3)
"""model visual question answering done"""

