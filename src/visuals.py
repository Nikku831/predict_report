import matplotlib.pyplot as plt
import os
from fpdf import FPDF
import pandas as pd
from datetime import datetime

class Visualizer:
    def generate_charts(self, df, ticker, forecast_df, currency_symbol="$"):
        """Creates a chart showing historical Price, SMA, and Prophet Forecast."""
        os.makedirs("output", exist_ok=True)
        chart_path = f"output/{ticker}_forecast_chart.png"
        
        plt.figure(figsize=(12, 6))
        
        # Plot Historical Data
        plt.plot(df.index, df['Close'], label='Historical Close Price', color='#3b82f6', linewidth=2)
        
        # Plot Forecast
        plt.plot(forecast_df['ds'], forecast_df['yhat'], label='Prophet Forecast', color='#10b981', linestyle='--')
        plt.fill_between(
            forecast_df['ds'], 
            forecast_df['yhat_lower'], 
            forecast_df['yhat_upper'], 
            color='#10b981', 
            alpha=0.2, 
            label='95% Confidence Interval'
        )
        
        plt.title(f"{ticker} - Price History and Prophet 30-Day Forecast", fontsize=16)
        plt.xlabel("Date")
        # Dynamic Currency Label
        plt.ylabel(f"Price ({currency_symbol})")
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(chart_path)
        plt.close()
        print(f" Prophet Chart saved to {chart_path}")
        return chart_path

class PDFGenerator(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'AI Equity Research Report', 0, 1, 'C')
        self.ln(5)

    def write_markdown(self, text):
        """
        Parses simple Markdown syntax for FPDF.
        """
        self.set_font('Arial', '', 11)
        
        # Sanitize text for FPDF (handle encoding for symbols like ₹)
        # We replace specific unicode chars that FPDF might struggle with if not using a unicode font
        # For basic FPDF, we might need to rely on 'Rs.' if the font doesn't support ₹, 
        # but modern environments often handle it or we accept the encoding replacement.
        try:
            # Try latin-1, if ₹ fails, it might be replaced or raise error. 
            # Ideally, we replace ₹ with 'Rs.' for PDF safety in basic FPDF
            text = text.replace('₹', 'Rs. ')
            text = text.encode('latin-1', 'replace').decode('latin-1')
        except:
            pass

        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                self.ln(2)
                continue
            
            if line.startswith('#'):
                clean_line = line.lstrip('#').strip()
                self.ln(3)
                self.set_font('Arial', 'B', 12)
                self.cell(0, 8, clean_line, 0, 1)
                self.set_font('Arial', '', 11)
                continue
                
            if line.startswith('- ') or line.startswith('* '):
                self.set_x(15)
                line = line[2:]
                self.write(5, '\x95 ')
            
            parts = line.split('**')
            for i, part in enumerate(parts):
                if i % 2 == 1:
                    self.set_font('Arial', 'B', 11)
                    self.write(5, part)
                    self.set_font('Arial', '', 11)
                else:
                    self.write(5, part)
            
            self.ln(6)

    def create_pdf(self, schema_data, report_text, chart_path):
        """Generates the final PDF using FPDF."""
        pdf = PDFGenerator()
        pdf.add_page()
        os.makedirs("output", exist_ok=True)
        pdf_filename = f"output/{schema_data['ticker']}_Report_FPDF.pdf"

        # Determine safe display symbol for PDF (Fpdf standard fonts often lack ₹ glyph)
        display_sym = "Rs." if schema_data['currency_symbol'] == "₹" else schema_data['currency_symbol']

        # --- Metadata Section ---
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, f"Ticker: {schema_data['ticker']}", 0, 1)
        
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 5, f"Date: {datetime.now().strftime('%Y-%m-%d')}", 0, 1)
        
        # Use display_sym for PDF compatibility
        pdf.cell(0, 5, f"Price: {display_sym}{schema_data['current_price']} | Trend: {schema_data['trend']}", 0, 1)
        pdf.cell(0, 5, f"Target: {display_sym}{schema_data['prediction_30d']:.2f}", 0, 1)
        pdf.ln(5)
        
        # --- Chart ---
        if chart_path and os.path.exists(chart_path):
            pdf.image(chart_path, x=10, w=190)
            pdf.ln(5)
        
        # --- Analyst Commentary ---
        pdf.write_markdown(report_text)
        
        pdf.output(pdf_filename)
        print(f" PDF Report saved: {pdf_filename}")
        return pdf_filename