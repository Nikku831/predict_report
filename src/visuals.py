import matplotlib.pyplot as plt
import os
from fpdf import FPDF
import pandas as pd
from datetime import datetime
import re

class Visualizer:
    def generate_charts(self, df, ticker, forecast_df, currency_symbol="$"):
        os.makedirs("output", exist_ok=True)
        chart_path = f"output/{ticker}_forecast_chart.png"
        
        plt.figure(figsize=(10, 5))
        plt.plot(df.index, df['Close'], label='Historical', color='#3b82f6', linewidth=2)
        plt.plot(forecast_df['ds'], forecast_df['yhat'], label='Forecast', color='#10b981', linestyle='--')
        plt.fill_between(forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'], color='#10b981', alpha=0.2)
        
        plt.title(f"{ticker} - Price History & AI Forecast")
        plt.xlabel("Date")
        plt.ylabel(f"Price ({currency_symbol})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(chart_path)
        plt.close()
        return chart_path

class PDFGenerator(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'AI Equity Research Report', 0, 1, 'C')
        self.ln(5)

    def write_markdown(self, text):
        self.set_font('Arial', '', 11)
        
        # 1. Basic Character Cleanup
        # (Note: Heavy cleanup is now done in main.py via delimiters)
        text = text.replace("’", "'").replace("“", '"').replace("”", '"').replace("–", "-")
        try:
            text = text.encode('latin-1', 'replace').decode('latin-1')
        except:
            pass
            
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                self.ln(2)
                continue
            
            # Skip raw table rows if they look like "| Metric | Value |"
            if line.startswith('|') and line.endswith('|'):
                continue 
            
            # Headers
            if line.startswith('#'):
                clean_line = line.lstrip('#').strip()
                self.ln(5)
                self.set_font('Arial', 'B', 12)
                self.set_text_color(30, 60, 138)
                self.cell(0, 8, clean_line, 0, 1)
                self.set_text_color(0, 0, 0)
                self.set_font('Arial', '', 11)
                continue
            
            # Bullets
            if line.startswith('- ') or line.startswith('* '):
                self.set_x(15)
                line = line[2:]
                self.write(5, chr(149) + ' ')
            
            # Bold Text Handling
            parts = re.split(r'(\*\*.*?\*\*)', line)
            for part in parts:
                if part.startswith('**') and part.endswith('**'):
                    self.set_font('Arial', 'B', 11)
                    self.write(5, part.strip('*'))
                    self.set_font('Arial', '', 11)
                else:
                    self.write(5, part)
            self.ln(6)

    def add_financial_dashboard(self, sentiment_data):
        """Adds a visual box with scores."""
        if not sentiment_data:
            sentiment_data = {}
            
        rec = sentiment_data.get('recommendation', {})
        analysis = sentiment_data.get('analysis', {})
        reasoning = rec.get('reasoning', {})
        
        rating = rec.get('rating', 'HOLD')
        fhi_score = analysis.get('fhi', {}).get('score', 'N/A')
        
        try:
            raw_sent = analysis.get('sentiment', {}).get('compound', 0)
            sent_score = f"{float(raw_sent):.2f}"
        except:
            sent_score = "0.00"
            
        risk = rec.get('risk_level') or reasoning.get('risk_level') or "Medium"
        if str(risk).upper() == "UNKNOWN": 
            risk = "High (Data Gap)"

        # Draw the Dashboard Box
        self.set_font('Arial', 'B', 12)
        self.set_text_color(30, 60, 138)
        self.cell(0, 8, "Financial Health & Sentiment Dashboard", 0, 1)
        self.set_text_color(0, 0, 0)
        self.ln(2)

        self.set_fill_color(240, 240, 240)
        self.rect(10, self.get_y(), 190, 30, 'F')
        
        self.set_y(self.get_y() + 5)
        self.set_font('Arial', 'B', 10)
        
        self.set_x(15)
        self.cell(45, 10, f"Recommendation: {rating}", 0, 0)
        self.cell(45, 10, f"FHI Score: {fhi_score}/100", 0, 0)
        self.cell(45, 10, f"Sentiment: {sent_score}", 0, 0)
        self.cell(45, 10, f"Risk Level: {str(risk).upper()}", 0, 1)
        self.ln(15)

    def create_crew_pdf(self, result_text, ticker, sentiment_data=None):
        pdf = PDFGenerator()
        pdf.add_page()
        os.makedirs("output", exist_ok=True)
        pdf_filename = f"output/{ticker}_Crew_Report.pdf"

        # 1. Header
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, f"Target Ticker: {ticker}", 0, 1)
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1)
        pdf.ln(5)
        
        # 2. Dashboard
        if sentiment_data:
            pdf.add_financial_dashboard(sentiment_data)

        # 3. Chart
        chart_path = f"output/{ticker}_forecast_chart.png"
        if os.path.exists(chart_path):
            pdf.set_font('Arial', 'B', 12)
            pdf.set_text_color(30, 60, 138)
            pdf.cell(0, 8, "30-Day Forecast Chart", 0, 1)
            pdf.set_text_color(0, 0, 0)
            pdf.image(chart_path, x=10, w=190)
            pdf.ln(5)
        
        # 4. Text
        pdf.write_markdown(result_text)
        pdf.output(pdf_filename)
        return pdf_filename