import os
import re
import json
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from urllib.parse import urlparse, parse_qs

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import yt_dlp
import requests
from google import generativeai as genai # Sử dụng SDK mới
import tempfile


# Configuration
app = Flask(__name__)
CORS(app)

# Environment variables
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

# Initialize Gemini client with new SDK syntax
genai.configure(api_key=GEMINI_API_KEY)

@dataclass
class VideoAnalysis:
    video_id: str
    title: str
    description: str
    thumbnail_url: str
    duration: int
    view_count: int
    like_count: int
    comment_count: int
    upload_date: str
    channel_name: str
    tags: List[str]
    
class YouTubeAnalyzer:
    def __init__(self):
        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
    
    def extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
            r'youtube\.com/v/([^&\n?#]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    async def get_video_info(self, url: str) -> Optional[VideoAnalysis]:
        """Extract video metadata using yt-dlp"""
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                return VideoAnalysis(
                    video_id=info.get('id', ''),
                    title=info.get('title', ''),
                    description=info.get('description', ''),
                    thumbnail_url=info.get('thumbnail', ''),
                    duration=info.get('duration', 0),
                    view_count=info.get('view_count', 0),
                    like_count=info.get('like_count', 0),
                    comment_count=info.get('comment_count', 0),
                    upload_date=info.get('upload_date', ''),
                    channel_name=info.get('uploader', ''),
                    tags=info.get('tags', [])
                )
        except Exception as e:
            print(f"Error extracting video info: {e}")
            return None
    
    async def download_video_segment(self, url: str, max_duration: int = 60) -> Optional[str]:
        """Download a video segment for analysis"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_path = temp_file.name
            
            ydl_opts = {
                'outtmpl': temp_path,
                'format': 'best[ext=mp4][height<=720]/best[ext=mp4]',
                'postprocessor_args': ['-ss', '0', '-t', str(max_duration)],
                'quiet': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            return temp_path
        except Exception as e:
            print(f"Error downloading video: {e}")
            return None

    async def analyze_with_gemini(self, video_info: VideoAnalysis, video_path: Optional[str] = None) -> Dict[str, Any]:
        """Analyze video using Gemini 1.5 Flash"""
        video_file_uploaded = None
        try:
            if video_path and os.path.exists(video_path):
                print(f"Uploading file: {video_path}")
                video_file_uploaded = genai.upload_file(path=video_path)
                while video_file_uploaded.state.name == "PROCESSING":
                    await asyncio.sleep(2)
                    video_file_uploaded = genai.get_file(video_file_uploaded.name)

                if video_file_uploaded.state.name == "FAILED":
                     raise ValueError("Video file processing failed.")
            
            prompt = f"""
            Bạn là chuyên gia phân tích YouTube. Hãy phân tích video này và đưa ra điểm số chi tiết:

            THÔNG TIN VIDEO:
            - Tiêu đề: {video_info.title}
            - Mô tả: {video_info.description[:500]}...
            - Kênh: {video_info.channel_name}
            - Tags: {', '.join(video_info.tags[:10])}
            - Lượt xem: {video_info.view_count:,}
            - Lượt thích: {video_info.like_count:,}

            PHÂN TÍCH YÊU CẦU:
            1. Đánh giá từng yếu tố (0-100 điểm): Thumbnail, Tiêu đề, Mô tả, Tags, Engagement, SEO tổng thể.
            2. Gợi ý cải thiện cụ thể (ưu tiên cao/trung/thấp).
            3. Dự đoán tiềm năng viral (0-100).

            Trả về định dạng JSON nghiêm ngặt theo cấu trúc sau:
            {{
                "overall_score": 0, "scores": {{"thumbnail": 0, "title": 0, "description": 0, "tags": 0, "engagement": 0, "seo": 0}},
                "improvements": [{{"category": "string", "priority": "high/medium/low", "suggestion": "string", "impact": "string"}}],
                "viral_potential": 0, "strengths": ["string"], "weaknesses": ["string"]
            }}
            """
            
            model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            
            contents = [prompt]
            if video_file_uploaded:
                contents.append(video_file_uploaded)

            response = await model.generate_content_async(contents)
            
            result_text = response.text.strip().replace('```json', '').replace('```', '')
            analysis_result = json.loads(result_text)
            
            return analysis_result
            
        except Exception as e:
            print(f"Error in Gemini analysis: {e}")
            return self.get_fallback_analysis()
        finally:
            if video_path and os.path.exists(video_path):
                os.unlink(video_path)
            if video_file_uploaded:
                genai.delete_file(video_file_uploaded.name)
    
    def get_fallback_analysis(self) -> Dict[str, Any]:
        """Fallback analysis if Gemini fails"""
        return {"error": "Failed to analyze with Gemini, providing fallback data.", "overall_score": 0, "scores": {}, "improvements": [], "viral_potential": 0, "strengths": [], "weaknesses": []}

analyzer = YouTubeAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
async def analyze_video():
    try:
        data = request.get_json()
        video_url = data.get('url')
        if not video_url: return jsonify({'error': 'URL is required'}), 400
        if not analyzer.extract_video_id(video_url): return jsonify({'error': 'Invalid YouTube URL'}), 400
        
        video_info = await analyzer.get_video_info(video_url)
        if not video_info: return jsonify({'error': 'Failed to extract video information'}), 400
        
        video_path = await analyzer.download_video_segment(video_url) if data.get('analyze_video_content') else None
        
        analysis_result = await analyzer.analyze_with_gemini(video_info, video_path)
        
        response_data = {'video_info': video_info.__dict__, 'analysis': analysis_result}
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Analysis error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

