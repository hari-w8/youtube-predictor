import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF
import tempfile
import os
import cv2
from PIL import Image
import io
import base64

class YouTubePredictor:
    def __init__(self):
        self.model = self._initialize_model()

    def _initialize_model(self):
        # This is a mock ML model - in real scenario, you'd train on actual data
        # For demonstration, we'll use rule-based predictions
        return "Mock Model"

    def predict_metrics(self, inputs, time_period="1 month"):
        # Mock prediction logic - replace with actual trained model
        subscribers = inputs['subscribers']
        duration_sec = inputs['duration_sec']
        description_length = inputs['description_length']
        video_quality = inputs['video_quality']

        # Base metrics calculation (mock logic)
        base_views = min(subscribers * 100, 1000000)  # Mock calculation
        quality_multiplier = {'480p': 0.8, '720p': 1.0, '1080p': 1.2, '4k': 1.5}.get(video_quality, 1.0)
        duration_multiplier = max(0.5, min(2.0, 120 / max(1, duration_sec)))

        # Time period multipliers
        time_multipliers = {
            "1 month": 1.0,
            "2 months": 1.8,
            "6 months": 3.5,
            "1 year": 6.0,
            "above 1 year": 8.0
        }

        time_multiplier = time_multipliers.get(time_period, 1.0)

        estimated_views = int(
            base_views * quality_multiplier * duration_multiplier * time_multiplier * np.random.uniform(0.8, 1.2))
        estimated_likes = int(estimated_views * 0.01 * np.random.uniform(0.8, 1.2))
        estimated_comments = int(estimated_views * 0.001 * np.random.uniform(0.8, 1.2))

        # Add some randomness for demo
        views_range = (int(estimated_views * 0.9), int(estimated_views * 1.1))
        # Add this at the end before return:
        video_score = self.calculate_video_score({
            'views_range': views_range,
            'likes': estimated_likes,
            'comments': estimated_comments
        }, inputs)
        return {
            'views_range': views_range,
            'likes': estimated_likes,
            'comments': estimated_comments,
            'comment_ratio': estimated_comments / max(1, estimated_likes),
            'time_period': time_period,
            'video_score': video_score  # ADD THIS LINE
        }

    def predict_for_all_periods(self, inputs):
        periods = ["1 month", "2 months", "6 months", "1 year", "above 1 year"]
        predictions = {}

        for period in periods:
            predictions[period] = self.predict_metrics(inputs, period)

        return predictions

    def suggest_best_time(self, category, current_time):
        # Mock best time suggestion based on category
        best_times = {
            'Anime': {'time': '19:00', 'day': 'Friday'},
            'Gaming': {'time': '20:00', 'day': 'Saturday'},
            'Education': {'time': '15:00', 'day': 'Wednesday'},
            'Music': {'time': '18:00', 'day': 'Friday'},
            'Vlog': {'time': '12:00', 'day': 'Sunday'}
        }

        default_time = {'time': '19:00', 'day': 'Friday'}
        suggested = best_times.get(category, default_time)

        # Calculate next occurrence of suggested day
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        current_day_index = datetime.now().weekday()
        suggested_day_index = days.index(suggested['day'])

        days_ahead = (suggested_day_index - current_day_index) % 7
        if days_ahead == 0:
            days_ahead = 7  # Next week

        next_date = datetime.now() + timedelta(days=days_ahead)

        return {
            'time': suggested['time'],
            'date': next_date.strftime('%d/%m/%Y'),
            'day': suggested['day']
        }

    def calculate_video_score(self, predictions, inputs):
        """Calculate overall video quality score (1-100)"""
        base_score = (
                predictions['views_range'][1] * 0.4 +
                predictions['likes'] * 0.3 +
                predictions['comments'] * 0.2 +
                min(inputs['description_length'] * 0.5, 10)  # Max 10 points for description
        )

        # Normalize to 0-100 scale
        normalized_score = min(100, base_score / 1000)

        # Quality multiplier
        quality_multiplier = {'480p': 0.7, '720p': 0.85, '1080p': 1.0, '4k': 1.15}.get(inputs['video_quality'], 1.0)

        final_score = int(normalized_score * quality_multiplier)
        return max(1, min(100, final_score))

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'YouTube Analytics Prediction Report', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(4)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()


def get_video_duration(video_path):
    """Extract video duration using OpenCV"""
    try:
        video = cv2.VideoCapture(video_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        video.release()
        return int(duration)
    except Exception as e:
        st.error(f"Error getting video duration: {str(e)}")
        return 0


def get_video_quality(video_path):
    """Extract video quality using OpenCV"""
    try:
        video = cv2.VideoCapture(video_path)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video.release()

        if width >= 3840 or height >= 2160:
            return "4k"
        elif width >= 1920 or height >= 1080:
            return "1080p"
        elif width >= 1280 or height >= 720:
            return "720p"
        else:
            return "480p"
    except Exception as e:
        st.error(f"Error getting video quality: {str(e)}")
        return "Unknown"


def create_visualizations(views_range, likes, comments):
    """Create bar chart and line graph"""
    # Bar chart
    metrics_df = pd.DataFrame({
        'Metric': ['Views', 'Likes', 'Comments'],
        'Count': [views_range[1], likes, comments]
    })

    bar_fig = px.bar(metrics_df, x='Metric', y='Count',
                     title='Predicted Engagement Metrics',
                     color='Metric')

    # Line graph (mock data for trend)
    days = list(range(1, 8))
    views_trend = [views_range[1] * i / 7 * np.random.uniform(0.9, 1.1) for i in range(1, 8)]
    likes_trend = [likes * i / 7 * np.random.uniform(0.9, 1.1) for i in range(1, 8)]

    line_fig = go.Figure()
    line_fig.add_trace(go.Scatter(x=days, y=views_trend, mode='lines+markers', name='Views'))
    line_fig.add_trace(go.Scatter(x=days, y=likes_trend, mode='lines+markers', name='Likes'))
    line_fig.update_layout(title='Predicted Growth Over 7 Days', xaxis_title='Days', yaxis_title='Count')

    return bar_fig, line_fig


def create_time_period_chart(all_predictions):
    """Create chart showing predictions across different time periods"""
    periods = list(all_predictions.keys())
    views = [all_predictions[period]['views_range'][1] for period in periods]
    likes = [all_predictions[period]['likes'] for period in periods]
    comments = [all_predictions[period]['comments'] for period in periods]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=periods, y=views, mode='lines+markers', name='Views', line=dict(width=4)))
    fig.add_trace(go.Scatter(x=periods, y=likes, mode='lines+markers', name='Likes', line=dict(width=4)))
    fig.add_trace(go.Scatter(x=periods, y=comments, mode='lines+markers', name='Comments', line=dict(width=4)))

    fig.update_layout(
        title='Predicted Growth Across Different Time Periods',
        xaxis_title='Time Period',
        yaxis_title='Count',
        hovermode='x unified'
    )

    return fig


def generate_pdf_report(username, video_title, all_predictions, selected_period, best_time):
    """Generate PDF report"""
    pdf = PDFReport()
    pdf.add_page()

    # User Information
    pdf.chapter_title('User Information')
    pdf.chapter_body(f'Username: {username}\nVideo Title: {video_title}')

    # Predictions for selected period
    selected_prediction = all_predictions[selected_period]
    pdf.chapter_title(f'Predicted Analytics ({selected_period})')
    pdf.chapter_body(
        f'Estimated Views: {selected_prediction["views_range"][0]:,} - {selected_prediction["views_range"][1]:,}\n'
        f'Estimated Likes: {selected_prediction["likes"]:,}\n'
        f'Estimated Comments: {selected_prediction["comments"]:,}\n'
        f'Comment Ratio: {selected_prediction["comment_ratio"]:.2f}'
    )
    # Add after predictions section:
    pdf.chapter_title('Video Quality Assessment')
    selected_prediction = all_predictions[selected_period]
    score = selected_prediction['video_score']

    if score >= 80:
        rating = "Excellent"
    elif score >= 60:
        rating = "Good"
    elif score >= 40:
        rating = "Average"
    else:
        rating = "Needs Improvement"

    pdf.chapter_body(
        f'Video Quality Score: {score}/100\n'
        f'Rating: {rating}\n'
        f'This score evaluates overall video potential based on predicted engagement and content quality.'
    )

    # All periods summary
    pdf.chapter_title('Summary for All Time Periods')
    summary_text = ""
    for period, pred in all_predictions.items():
        summary_text += f'{period}: {pred["views_range"][1]:,} views, {pred["likes"]:,} likes, {pred["comments"]:,} comments\n'
    pdf.chapter_body(summary_text)

    # Best Upload Time
    pdf.chapter_title('Recommended Upload Schedule')
    pdf.chapter_body(
        f'Best Time: {best_time["time"]}\n'
        f'Best Date: {best_time["date"]}\n'
        f'Best Day: {best_time["day"]}'
    )

    # Analytics note
    pdf.chapter_title('Analytics Charts')
    pdf.chapter_body(
        'The report includes bar charts and line graphs showing predicted engagement metrics and growth trends across different time periods.')

    return pdf


def main():
    st.set_page_config(page_title="YouTube Analytics Predictor", layout="wide")

    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = ""
    if 'predictor' not in st.session_state:
        st.session_state.predictor = YouTubePredictor()
    if 'predictions_made' not in st.session_state:
        st.session_state.predictions_made = False
    if 'all_predictions' not in st.session_state:
        st.session_state.all_predictions = {}
    if 'selected_period' not in st.session_state:
        st.session_state.selected_period = "1 month"
    if 'best_time' not in st.session_state:
        st.session_state.best_time = {
            'time': '19:00',
            'date': datetime.now().strftime('%d/%m/%Y'),
            'day': 'Friday'
        }

    # Login Section
    if not st.session_state.authenticated:
        st.title("YouTube Analytics Predictor - Login")

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")

            if login_button:
                # Simple authentication - in production, use proper authentication
                if username and password:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Please enter both username and password")
        return

    # Main Application
    st.title("YouTube Analytics Predictor")
    st.sidebar.header(f"Welcome, {st.session_state.username}!")

    # File upload section
    st.header("Upload Your Video")
    uploaded_file = st.file_uploader("Choose a video file",
                                     type=['mp4', 'mov', 'avi', 'wmv', 'webm'])

    if uploaded_file is not None:
        # Validate file format
        allowed_formats = ['mp4', 'mov', 'avi', 'wmv', 'webm']
        file_extension = uploaded_file.name.split('.')[-1].lower()

        if file_extension not in allowed_formats:
            st.error(f"Error: Invalid file format. Please upload one of: {', '.join(allowed_formats)}")
        else:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                video_path = tmp_file.name

            # Display video with custom size
            st.subheader("Uploaded Video Preview")

            # CSS to style the video player
            st.markdown("""
            <style>
                .stVideo {
                    max-width: 200vh;
                    height: 550px;
                }
                .stVideo video {
                    width: 100% !important;
                    height: 230px !important;
                    object-fit: contain;
                }
            </style>
            """, unsafe_allow_html=True)

            st.video(uploaded_file.getvalue())

            # Extract video properties
            duration_seconds = get_video_duration(video_path)
            video_quality = get_video_quality(video_path)

            # Input form
            st.header("Video Details")
            with st.form("video_details"):
                video_title = st.text_input("Video Title", value="Akaza Tamil - AMV edit - demon slayer")
                category = st.text_input("Category", value="Anime, Edits")
                tags = st.text_area("Tags (separate with commas)", value="#anime #new #animeedits #tamil #status")
                subscribers = st.number_input("Subscribers", min_value=0, value=1000)

                col1, col2 = st.columns(2)
                with col1:
                    upload_time = st.time_input("Uploading Time", value=datetime.strptime("20:13", "%H:%M").time())
                with col2:
                    upload_date = st.date_input("Uploading Date", value=datetime(2025, 9, 23))
                    upload_day = upload_date.strftime("%A")

                description = st.text_area("Description",
                                           value="Akaza's love is a tragic and twisted cornerstone of his character in Demon Slayer. Before becoming a demon, he was Hakuji, a man deeply devoted to his fiancée, Koyuki. His love for her and her father was pure and all-consuming. Their murder shattered him, driving him to become the demon Akaza.")

                description_length = len(description.split())

                # Time period selection
                time_period = st.selectbox(
                    "Select Time Period for Prediction",
                    ["1 month", "2 months", "6 months", "1 year", "above 1 year"],
                    index=0
                )

                # Display auto-extracted information
                st.info(f"Auto-detected Duration: {duration_seconds} seconds")
                st.info(f"Auto-detected Quality: {video_quality}")
                st.info(f"Description Length: {description_length} words")

                predict_button = st.form_submit_button("Predict Analytics")

            if predict_button:
                # Prepare inputs for prediction
                inputs = {
                    'subscribers': subscribers,
                    'duration_sec': duration_seconds,
                    'description_length': description_length,
                    'video_quality': video_quality,
                    'category': category,
                    'upload_time': upload_time,
                    'upload_date': upload_date
                }

                # Get predictions for all time periods
                with st.spinner("Analyzing your video..."):
                    st.session_state.all_predictions = st.session_state.predictor.predict_for_all_periods(inputs)
                    st.session_state.best_time = st.session_state.predictor.suggest_best_time(category, upload_time)
                    st.session_state.predictions_made = True
                    st.session_state.selected_period = time_period
                    st.session_state.current_video_title = video_title
                    st.session_state.current_category = category

            # Display results if predictions were made
            if st.session_state.predictions_made:
                st.header("Prediction Results")

                # Time period selector for current view
                selected_period = st.selectbox(
                    "View predictions for:",
                    list(st.session_state.all_predictions.keys()),
                    index=list(st.session_state.all_predictions.keys()).index(st.session_state.selected_period)
                )

                predictions = st.session_state.all_predictions[selected_period]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Estimated Views",
                              f"{predictions['views_range'][0]:,} - {predictions['views_range'][1]:,}",
                              help=f"Predicted views over {selected_period}")
                with col2:
                    st.metric("Estimated Likes", f"{predictions['likes']:,}",
                              help=f"Predicted likes over {selected_period}")
                with col3:
                    st.metric("Estimated Comments", f"{predictions['comments']:,}",
                              help=f"Predicted comments over {selected_period}")

                st.info(f"Comment Ratio: {predictions['comment_ratio']:.2f}")
                st.metric("Video Quality Score", f"{predictions['video_score']}/100")
                # Add score interpretation
                score = predictions['video_score']
                if score >= 80:
                    st.success("🎯 Excellent! This video has high potential!")
                elif score >= 60:
                    st.info("📈 Good! This video should perform well!")
                elif score >= 40:
                    st.warning("⚠️ Average! Consider optimizing your content.")
                else:
                    st.error("🔧 Needs improvement! Review video elements.")

                # Display all periods in a table
                st.subheader("Predictions for All Time Periods")
                period_data = []
                for period, pred in st.session_state.all_predictions.items():
                    # In the period_data table, add:
                    period_data.append({
                        'Time Period': period,
                        'Views': f"{pred['views_range'][1]:,}",
                        'Likes': f"{pred['likes']:,}",
                        'Comments': f"{pred['comments']:,}",
                        'Comment Ratio': f"{pred['comment_ratio']:.2f}",
                        'Video Score': f"{pred['video_score']}/100"  # ADD THIS
                    })

                st.table(pd.DataFrame(period_data))

                # Best upload time suggestion
                st.subheader("Recommended Upload Schedule")
                st.success(
                    f"Best Upload Time: {st.session_state.best_time['time']}, {st.session_state.best_time['date']} ({st.session_state.best_time['day']})")

                # Visualizations
                st.subheader("Analytics Visualizations")
                bar_fig, line_fig = create_visualizations(
                    predictions['views_range'],
                    predictions['likes'],
                    predictions['comments']
                )

                st.plotly_chart(bar_fig, use_container_width=True)
                st.plotly_chart(line_fig, use_container_width=True)

                # Time period comparison chart
                st.subheader("Growth Across Time Periods")
                time_period_fig = create_time_period_chart(st.session_state.all_predictions)
                st.plotly_chart(time_period_fig, use_container_width=True)

                # PDF Report Generation
                st.subheader("Download Report")

                # Create PDF when button is clicked
                if st.button("Generate PDF Report", key="generate_pdf_btn"):  # ADD KEY HERE
                    if st.session_state.predictions_made:
                        try:
                            pdf = generate_pdf_report(
                                st.session_state.username,
                                st.session_state.current_video_title,
                                st.session_state.all_predictions,
                                selected_period,
                                st.session_state.best_time,
                            )

                            # Generate PDF bytes
                            pdf_bytes = bytes(pdf.output(dest='S'))


                            # Create download button with unique key
                            st.download_button(
                                label="Download PDF Report",
                                data=pdf_bytes,
                                file_name=f"youtube_analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                mime="application/pdf",
                                key=f"pdf_download_{datetime.now().strftime('%Y%m%d_%H%M%S')}"  # UNIQUE KEY
                            )

                        except Exception as e:
                            st.error(f"Error generating PDF: {str(e)}")
                    else:
                        st.warning("Please generate predictions first by clicking 'Predict Analytics'")
            # Clean up temporary file
            try:
                os.unlink(video_path)
            except:
                pass

    # Add this section before the logout button in main() function

    # Comment Section
    st.header("💬 Community Comments")
    st.info("Share your thoughts and experiences with other creators!")

    # Display existing comments from session state
    if 'comments' not in st.session_state:
        st.session_state.comments = []

    # Show existing comments
    if st.session_state.comments:
        st.subheader("Recent Comments")
        for i, comment in enumerate(st.session_state.comments):
            with st.container():
                st.markdown(f"""
                <div style='background-color: black ; padding: 15px; border-radius: 10px; margin: 10px 0;'>
                    <strong>👤 {comment['username']}</strong> 
                    <small style='color: #666;'>{comment['timestamp']}</small>
                    <br>
                    {comment['text']}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.write("No comments yet. Be the first to share your thoughts!")

    # Add new comment form
    with st.form("comment_form", clear_on_submit=True):
        comment_text = st.text_area("Add your comment", placeholder="Share your experience, tips, or feedback...")
        submit_comment = st.form_submit_button("Post Comment")

        if submit_comment and comment_text:
            new_comment = {
                'username': st.session_state.username,
                'text': comment_text,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            st.session_state.comments.insert(0, new_comment)  # Add to beginning
            st.success("Comment posted!")
            st.rerun()

    # Logout button
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = ""
        st.session_state.predictions_made = False
        st.session_state.all_predictions = {}
        st.session_state.best_time = {
            'time': '19:00',
            'date': datetime.now().strftime('%d/%m/%Y'),
            'day': 'Friday'
        }
        st.rerun()


if __name__ == "__main__":
    main()
