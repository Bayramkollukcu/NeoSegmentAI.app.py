import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# SAYFA YAPILANDIRMASI
# ------------------------------
st.set_page_config(page_title="Müşteri 360° - Segmentasyon & Tahmin", layout="wide")
st.title("🛍️ E-ticaret Müşteri Analitik Prototipi")
st.markdown("Ralph Lauren 4D Modeli (Derinlik, Dinamiklik, Arzu Edilirlik, Dağıtım) esinlidir.")

# ------------------------------
# 1. VERİ OLUŞTURMA (SENTETİK)
# ------------------------------
@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 500

    customer_id = np.arange(1, n+1)
    recency_days = np.random.exponential(scale=15, size=n).astype(int).clip(0,90)
    frequency = np.random.poisson(lam=8, size=n) + 1
    avg_basket = np.random.gamma(2,100, size=n).astype(int) + 50
    monetary_total = (frequency * avg_basket).astype(int).clip(100,50000)
    last_30_days_visits = np.random.poisson(lam=5, size=n).clip(0,30)
    wishlist_count = np.random.poisson(lam=2, size=n).clip(0,15)
    preferred_channel = np.random.choice([0,1,2], size=n, p=[0.5,0.4,0.1])
    age = np.random.normal(35,10, size=n).astype(int).clip(18,70)
    discount_sensitivity = np.random.beta(2,5, size=n)
    avg_discount_used = (discount_sensitivity*50).astype(int).clip(0,50)

    # Hedef: next_purchase_30d
    log_odds = (-0.05*recency_days + 0.1*frequency + 0.005*(monetary_total/1000) +
                0.08*last_30_days_visits + 0.15*wishlist_count + 0.01*avg_discount_used -0.005*age -1.5)
    prob_next = 1/(1+np.exp(-log_odds))
    next_purchase_30d = (np.random.rand(n) < prob_next).astype(int)

    df = pd.DataFrame({
        'customer_id': customer_id,
        'recency_days': recency_days,
        'frequency': frequency,
        'monetary_total': monetary_total,
        'last_30_days_visits': last_30_days_visits,
        'wishlist_count': wishlist_count,
        'preferred_channel': preferred_channel,
        'age': age,
        'avg_discount_used': avg_discount_used,
        'next_purchase_30d': next_purchase_30d,
        'discount_sensitivity': discount_sensitivity
    })
    return df

df = generate_data()

# ------------------------------
# 2. MÜŞTERİ SEGMENTASYONU (K-Means)
# ------------------------------
features_seg = ['recency_days', 'frequency', 'monetary_total', 'last_30_days_visits', 'wishlist_count']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features_seg])

# Elbow grafiği
inertia = [KMeans(k, random_state=42, n_init=10).fit(X_scaled).inertia_ for k in range(2,8)]
fig_elbow, ax = plt.subplots()
ax.plot(range(2,8), inertia, marker='o')
ax.set_title('Elbow Yöntemi (Optimum Küme Sayısı)')
ax.set_xlabel('Küme Sayısı')
ax.set_ylabel('Inertia')
st.pyplot(fig_elbow)

# 4 küme ile segmentasyon
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['segment'] = kmeans.fit_predict(X_scaled)

# Segment isimlendirme (monetary_total ortalamasına göre)
centers = df.groupby('segment')['monetary_total'].mean().sort_values(ascending=False)
seg_names = {}
for i, seg in enumerate(centers.index):
    if i == 0:
        seg_names[seg] = 'Premium Sadık'
    elif i == 1:
        seg_names[seg] = 'Aktif Orta Sınıf'
    elif i == 2:
        seg_names[seg] = 'Fırsatçı İndirim Avcısı'
    else:
        seg_names[seg] = 'Riskli / Uyuyan'
df['segment_name'] = df['segment'].map(seg_names)

# PCA görselleştirme
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df['pca1'], df['pca2'] = pca_result[:,0], pca_result[:,1]

fig_pca, ax = plt.subplots(figsize=(8,5))
sns.scatterplot(data=df, x='pca1', y='pca2', hue='segment_name', palette='Set2', ax=ax)
ax.set_title('Segmentler (PCA Projeksiyonu)')
st.pyplot(fig_pca)

# ------------------------------
# 3. NEXT PURCHASE MODELİ (Random Forest)
# ------------------------------
feature_cols = ['recency_days', 'frequency', 'monetary_total', 'last_30_days_visits',
                'wishlist_count', 'preferred_channel', 'age', 'avg_discount_used']
X_model = df[feature_cols]
y_model = df['next_purchase_30d']

rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X_model, y_model, scoring='roc_auc', cv=cv)
auc_mean = cv_scores.mean()

# Tüm veri ile eğit (demo için)
rf.fit(X_model, y_model)
df['next_purchase_prob'] = rf.predict_proba(X_model)[:,1]

# Özellik önemleri
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
fig_imp, ax = plt.subplots()
importances.plot(kind='bar', ax=ax)
ax.set_title('Özellik Önemleri')
ax.set_ylabel('Önem')
st.pyplot(fig_imp)

# ------------------------------
# 4. KİŞİSELLEŞTİRİLMİŞ ÖNERİLER (Müşteri Seçimi)
# ------------------------------
st.markdown("---")
st.header("🔍 Müşteri Bazlı Analiz ve Öneriler")

customer_ids = df['customer_id'].tolist()
selected_id = st.selectbox("Bir müşteri ID seçin:", customer_ids)
cust = df[df['customer_id'] == selected_id].iloc[0]

col1, col2 = st.columns(2)
with col1:
    st.subheader("📋 Müşteri Profili")
    st.metric("Segment", cust['segment_name'])
    st.metric("Next Purchase Olasılığı", f"{cust['next_purchase_prob']:.0%}")
    st.metric("Son Alışveriş (gün)", f"{cust['recency_days']} gün")
    st.metric("Toplam Harcama (TL)", f"{cust['monetary_total']:,.0f}")
    st.metric("Wishlist Adedi", cust['wishlist_count'])
    st.metric("İndirim Duyarlılığı", f"{cust['discount_sensitivity']:.0%}")

with col2:
    st.subheader("🎯 Kişisel Öneriler")
    # Ürün önerisi (segment bazlı manuel)
    seg_product_map = {
        'Premium Sadık': 'Polo T-shirt (Premium Koleksiyon)',
        'Aktif Orta Sınıf': 'Classic Fit Oxford Gömlek',
        'Fırsatçı İndirim Avcısı': 'İndirimli Kapüşonlu Sweatshirt',
        'Riskli / Uyuyan': 'Sizi özledik! Özel %15 indirim kodu'
    }
    rec_product = seg_product_map.get(cust['segment_name'], 'Polo T-shirt')
    st.info(f"📦 **En İyi Sonraki Ürün:** {rec_product}")

    # Kanal önerisi
    channel_map = {0: 'Web sitesi', 1: 'Mobil App', 2: 'Mağaza'}
    st.success(f"📱 **Önerilen Kanal:** {channel_map[cust['preferred_channel']]}")

    # Zaman önerisi (recency'ye göre basit)
    if cust['recency_days'] < 7:
        hour = 10
    elif cust['recency_days'] < 30:
        hour = 14
    else:
        hour = 19
    st.warning(f"⏰ **En Uygun Zaman:** {hour}:00 - {hour+1}:00 arası")

    # İndirim stratejisi
    if cust['discount_sensitivity'] > 0.7:
        discount_str = "Yüksek indirim (%20 kupon) gönder"
    elif cust['discount_sensitivity'] > 0.4:
        discount_str = "Sepette %10 indirim teklifi"
    else:
        discount_str = "Özel koleksiyon tanıtımı, indirim yok"
    st.markdown(f"🏷️ **İndirim Stratejisi:** {discount_str}")

# ------------------------------
# 5. TOPLU PERFORMANS ÖZETİ
# ------------------------------
st.markdown("---")
st.header("📊 Model Performans Özeti")
st.metric("Cross-Validation ROC AUC (5-fold)", f"{auc_mean:.3f}")
st.caption("ROC AUC 0.70 üzeri iyi, 0.80 üzeri mükemmel kabul edilir.")

# Segment dağılımı
fig_seg, ax = plt.subplots()
df['segment_name'].value_counts().plot(kind='bar', ax=ax, color='lightblue')
ax.set_title('Segment Dağılımı')
ax.set_xlabel('Segment')
ax.set_ylabel('Müşteri Sayısı')
st.pyplot(fig_seg)

st.success("✅ Prototip başarıyla çalışıyor! Her müşteri için öneriler ve tahminler üretebilirsiniz.")
