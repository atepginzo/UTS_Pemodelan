from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Agar tidak error GUI di server
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# --- FUNGSI 1: LOAD DATA ---
def load_data():
    try:
        # Membaca file lokal (pastikan inventaris.csv ada di folder yang sama)
        df = pd.read_csv('inventaris.csv')
        
        # Deteksi format tanggal
        try:
            df['Order Date'] = pd.to_datetime(df['Order Date'], format='%d/%m/%Y')
        except:
            df['Order Date'] = pd.to_datetime(df['Order Date'], format='%m/%d/%Y')
            
        # Agregasi harian
        daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()
        daily_sales = daily_sales.set_index('Order Date')
        full_range = pd.date_range(start=daily_sales.index.min(), end=daily_sales.index.max(), freq='D')
        daily_sales = daily_sales.reindex(full_range, fill_value=0)
        
        return daily_sales['Sales'].values
    except Exception as e:
        print(f"Error: {e}")
        return None

# --- FUNGSI 2: LOGIKA SIMULASI (Sama persis dengan Colab) ---
def run_simulation(inventaris_awal, rop, target_inv, lead_time, qty_tetap, scenario_mode, shock_mult, shock_lead):
    
    data_permintaan = load_data()
    if data_permintaan is None:
        return None, None, None

    # Copy data
    demand_data = np.copy(data_permintaan)
    total_hari = len(demand_data)
    shock_index = -1
    
    # Logika Skenario
    if scenario_mode == "demand_shock":
        shock_index = np.argmax(demand_data)
        demand_data[shock_index] = demand_data[shock_index] * shock_mult
        
    current_lead_time = lead_time
    if scenario_mode == "supply_shock":
        current_lead_time = shock_lead

    # Loop Simulasi
    level_inv = inventaris_awal
    is_ordering = False
    arrival_day = -1
    
    hist_inv = []
    hist_stockout = []
    
    for day in range(total_hari):
        demand = demand_data[day]
        
        # Barang Masuk
        inflow = 0
        if is_ordering and day == arrival_day:
            inflow = qty_tetap
            is_ordering = False
            
        level_inv += inflow
        
        # Penuhi Permintaan
        stockout = 0
        if level_inv >= demand:
            level_inv -= demand
        else:
            stockout = demand - level_inv
            level_inv = 0
            
        # Cek ROP
        if level_inv < rop and not is_ordering:
            is_ordering = True
            arrival_day = day + current_lead_time
            
        hist_inv.append(level_inv)
        hist_stockout.append(stockout)
        
    return hist_inv, hist_stockout, shock_index

# --- ROUTE FLASK ---
@app.route('/', methods=['GET', 'POST'])
def index():
    plot_url = None
    results = None
    
    # Nilai default untuk form
    params = {
        'inv_awal': 80000, 'rop': 30000, 'target': 100000,
        'lead': 7, 'qty': 50000, 'scen': 'normal',
        'mult': 10, 's_lead': 14
    }

    if request.method == 'POST':
        # Ambil input dari Form HTML
        params['inv_awal'] = int(request.form.get('inv_awal'))
        params['rop'] = int(request.form.get('rop'))
        params['target'] = int(request.form.get('target'))
        params['lead'] = int(request.form.get('lead'))
        params['qty'] = int(request.form.get('qty'))
        params['scen'] = request.form.get('scen')
        params['mult'] = float(request.form.get('mult'))
        params['s_lead'] = int(request.form.get('s_lead'))

        # Jalankan Simulasi
        hist_inv, hist_stockout, s_idx = run_simulation(
            params['inv_awal'], params['rop'], params['target'], 
            params['lead'], params['qty'], params['scen'], 
            params['mult'], params['s_lead']
        )

        if hist_inv:
            # Hitung Statistik
            total_loss = sum(hist_stockout)
            days_stockout = sum(1 for x in hist_stockout if x > 0)
            
            results = {
                'loss': f"{total_loss:,.2f}",
                'days': days_stockout
            }

            # Buat Grafik dengan Matplotlib
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            
            # Grafik 1: Inventaris
            ax1.plot(hist_inv, label='Level Inventaris', color='blue')
            ax1.axhline(y=params['rop'], color='red', linestyle='--', label='ROP')
            if params['scen'] == 'demand_shock':
                ax1.annotate('SHOCK!', xy=(s_idx, hist_inv[s_idx]), 
                             xytext=(s_idx, hist_inv[s_idx]+20000), arrowprops=dict(facecolor='red', shrink=0.05))
            ax1.set_title(f"Level Inventaris (Skenario: {params['scen']})")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Grafik 2: Stockout
            ax2.bar(range(len(hist_stockout)), hist_stockout, color='orange')
            ax2.set_title("Kerugian (Stock-out)")
            ax2.set_xlabel("Hari ke-")
            
            plt.tight_layout()
            
            # Konversi Grafik ke Gambar Base64 agar bisa tampil di HTML
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()

    return render_template('index.html', plot_url=plot_url, results=results, p=params)

if __name__ == '__main__':
    app.run(debug=True)