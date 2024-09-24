import torch
from flask import Flask, request, jsonify, send_file
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
import os

# Initialize the pipeline
pipe = DiffusionPipeline.from_pretrained("damo-vilab/text-to-video-ms-1.7b", torch_dtype=torch.float16, variant="fp16")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# Initialize Flask app
app = Flask(__name__)

# POST endpoint for generating video
@app.route('/generate-video', methods=['POST'])
def generate_video():
    data = request.get_json()
    prompt = data.get('prompt')
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    # Generate video frames
    video_frames = pipe(prompt, num_inference_steps=25).frames
    
    # Export frames to video
    video_path = export_to_video(video_frames)
    
    # Send the video file back to the frontend
    return send_file(video_path, mimetype='video/mp4')

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
