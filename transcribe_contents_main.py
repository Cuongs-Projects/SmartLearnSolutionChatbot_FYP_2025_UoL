#!/usr/bin/env python3
import json
import os
#-------------------------------------------------------------------
#for video-to-text
import whisper
import torch
#from langchain.text_splitter import RecursiveCharacterTextSplitter

#!pip3 install langchain torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
# !pip3 install -U openai-whisper
#-------------------------------------------------------------------
#for pdf-to-text and img-to-text
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

# !pip3 install pytesseract transformers Pillow
#-------------------------------------------------------------------

json_template = {
    "id": None,
    "source_type": None,
    "source_identifier": None,
    "page_number":None,
    "startsec": None,
    "endsec":None,
    "text": None,
    "visual_description": None
}
base_dir = 'PADBRC'
teaching_contents = 'lectures_and_papers'
transcribed_dir = 'transcribed_contents'
contents_sources = 'contents_sources'
contents_sources_dir = os.path.join(contents_sources,'contents.json')

class TC_SML():
    #------------GENERAL----------------------
    @staticmethod
    def save_to_file(base_dir,output_dir,json_file,file):
        json_filename = os.path.splitext(file)[0] + ".json"
        file_dir = os.path.join(base_dir,output_dir,json_filename)
        with open(file_dir, "w", encoding="utf-8") as f:
            json.dump(json_file, f, indent=2, ensure_ascii=False)
        print(f"Segment details saved to {file_dir}")
    #-----------------------------------------
    #----------------VIDEO--------------------
    #-----------------------------------------
    @staticmethod
    def setup_whisper(model_type):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            model = whisper.load_model(model_type,device=device)
            print("Whisper model loaded successfully.")
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            if "CUDA" in str(e) and not torch.cuda.is_available():
                print("PyTorch cannot access your GPU. Ensure NVIDIA drivers are up to date and PyTorch was installed with CUDA support if you have a GPU.")
                exit()
            else:
                print("Make sure you have PyTorch installed correctly, and a GPU if you specified 'cuda'.")
                exit()
        return model
    
    @staticmethod
    def transcribe_the_video(video_path,model):
        print(f"Starting transcription of {video_path}...")
        try:
            result = model.transcribe(video_path, language="vi", verbose=False)
        except FileNotFoundError:
            print(f"Error: Audio file not found at {video_path}")
            exit()
        except Exception as e:
            print(f"An error occurred during transcription: {e}")
            exit()
        
        print(f"{video_path} is transcribed")
        return result['segments']

    @staticmethod
    def format_the_transcribed_video(segments,video_title,json_template,source_dir,file_name):
        with open(source_dir, 'r') as file:
            sources = json.load(file)
        ind = 0
        json_output = []
        for segment in segments:
            temp_json = json_template.copy()
            temp_json['id'] = file_name+"_"+str(ind)
            temp_json['source_type'] = 'video'
            temp_json['source_identifier'] = sources[video_title]
            temp_json['startsec'] = segment['start']
            temp_json['endsec'] = segment['end']
            temp_json['text'] = segment['text']
            json_output.append(temp_json)
            ind+=1
        print("Formatted into predefined json format")
        return json_output
    #-----------------------------------------
    #-----------------PDF---------------------
    #-----------------------------------------
    @staticmethod
    def transcribe_the_pdf(pdf_pages_as_images,pdf):
        images = pdf_pages_as_images
        full_text = ""
        for i, img in enumerate(images):
            page_text = pytesseract.image_to_string(img, lang="eng+vie")
            full_text += page_text
            full_text += "\n--- Page Break ---\n"
        print(f"Transcribed {pdf} into raw text")
        return full_text

    @staticmethod
    def format_transcribed_to_array_pdf(text):
        unprocessed_array = text.split('\n')
        output = []
        for i in range(len(unprocessed_array)):
            if unprocessed_array[i] != "":
                output.append(unprocessed_array[i])
        print(f"Transcribed raw text is now an array")
        return output

    @staticmethod
    def format_the_transcribed_pdf(text_array,pdf_title,json_template,source_dir,file_name):
        with open(source_dir, 'r') as file:
            sources = json.load(file)
        ind = 0
        page_no = 1
        json_output = []
        for text in text_array:
            if text != "--- Page Break ---":
                temp_json = json_template.copy()
                temp_json['id'] = file_name+"_"+str(ind)
                temp_json['source_type'] = 'pdf'
                temp_json['source_identifier'] = sources[pdf_title]
                temp_json['text'] = text
                temp_json['page_number'] = page_no
                json_output.append(temp_json)
                ind+=1
            elif text == "--- Page Break ---":
                page_no +=1
                
        print("Formatted into predefined json format")
        return json_output
    #-----------------------------------------
    #------------IMAGES/INSTRUCTIONS----------
    #-----------------------------------------
    @staticmethod
    def format_transcribed_to_array_instruction(text):
        lines = text.split('\n')
        output = []
        for i in range(len(lines)):
            if lines[i] != "":
                output.append(lines[i])
        return output

    @staticmethod
    def format_the_transcribed_instruction(text_array,instruction_title,json_template,source_dir,file_name):
        with open(source_dir, 'r') as file:
            sources = json.load(file)
        ind = 0
        json_output = []
        for text in text_array:
            temp_json = json_template.copy()
            temp_json['id'] = file_name+"_"+str(ind)
            temp_json['source_type'] = 'instruction_img'
            temp_json['source_identifier'] = sources[instruction_title]
            temp_json['text'] = text
            json_output.append(temp_json)
            ind+=1
        print("Formatted into predefined json format")
        return json_output
    
    #-----------------------------------------
    #-------------------MAIN------------------
    #-----------------------------------------
    @staticmethod
    def master_lecture_contents_to_json(base_dir,content_dir,contents_sources_dir,output_dir):
        contents_file_path = os.path.join(base_dir,content_dir)
        entries = os.listdir(contents_file_path)
        model  = TC_SML.setup_whisper("small")
        for material in entries:
            print(f"---Processing {material}")
            content_path = os.path.join(contents_file_path,material)
            if material.endswith(".mp4"):
                segments = TC_SML.transcribe_the_video(video_path=content_path, model=model)
                json_output = TC_SML.format_the_transcribed_video(segments=segments,video_title=material,
                                                            json_template=json_template,
                                                            source_dir=os.path.join(base_dir,contents_sources_dir),
                                                            file_name=os.path.splitext(material)[0])
                TC_SML.save_to_file(base_dir=base_dir,output_dir=output_dir,json_file=json_output,file = material)

            elif material.endswith(".pdf"):
                pdf_pages_as_images = convert_from_path(content_path)
                transcribed_text = TC_SML.transcribe_the_pdf(pdf_pages_as_images=pdf_pages_as_images,pdf = material)
                text_array = TC_SML.format_transcribed_to_array_pdf(transcribed_text)
                json_output = TC_SML.format_the_transcribed_pdf(text_array=text_array,pdf_title=material,
                                                        json_template=json_template,
                                                        source_dir=os.path.join(base_dir,contents_sources_dir),
                                                        file_name=os.path.splitext(material)[0])
                TC_SML.save_to_file(base_dir=base_dir,output_dir=output_dir,json_file=json_output, file = material)

            elif material.endswith(".png"):
                extracted_text = pytesseract.image_to_string(content_path, lang="eng+vie")
                text_array = TC_SML.format_transcribed_to_array_instruction(text=extracted_text)
                json_output = TC_SML.format_the_transcribed_instruction(text_array=text_array,instruction_title=material,
                                                        json_template=json_template,
                                                        source_dir=os.path.join(base_dir,contents_sources_dir),
                                                        file_name=os.path.splitext(material)[0])
                TC_SML.save_to_file(base_dir=base_dir,output_dir=output_dir,json_file=json_output, file = material)


if __name__ == "__main__":
    TC_SML.master_lecture_contents_to_json(base_dir=base_dir,
                                               content_dir=teaching_contents,
                                               contents_sources_dir=contents_sources_dir,
                                               output_dir=transcribed_dir)