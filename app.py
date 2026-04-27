import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# SAYFA YAPILANDIRMASI
# ------------------------------
st.set_page_config(page_title="Müşteri 360° - Segmentasyon & Tahmin", layout="wide")
st.title("🛍️ E-ticaret Müşteri Analitik Prototipi")
st.markdown("Ralph Lauren 4D Modeli (Derinlik, Dinamiklik, Arzu Edilirlik, Dağıtım) esinlidir.")

# ------------------------------
# 1. VERİ OLUŞTURMA (SENTETİK) - Geçmiş kategoriler ve hedef kategori dahil
# ------------------------------
@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 500

    # Müşteri temel özellikleri
    customer_id = np.arange(1, n+1)
    recency_days = np.random.exponential(scale=15, size=n).astype(int).clip(0,90)
    frequency = np.random.poisson(lam=8, size=n) + 1
    avg_basket = np.random.gamma(2,100, size=n).astype(int) + 50
    monetary_total = (frequency * avg_basket).astype(int).clip(100,50000)
    last_30_days_visits = np.random.poisson(lam=5, size=n).clip(0,30)
    wishlist_count = np.random.poisson(lam=2, size=n).clip(0,15)
    preferred_channel = np.random.choice([0,1,2], size=n, p=[0.5,0.4,0.1])
    
    # Ay tercihi
    month_probs = [0.06,0.06,0.07,0.08,0.09,0.12,0.13,0.12,0.09,0.07,0.06,0.05]
    preferred_month = np.random.choice(range(1,13), size=n, p=month_probs)
    
    age = np.random.normal(35,10, size=n).astype(int).clip(18,70)
    discount_sensitivity = np.random.beta(2,5, size=n)
    avg_discount_used = (discount_sensitivity*50).astype(int).clip(0,50)

    # ---------- Geçmiş alışveriş kategorileri ve hedef kategori ----------
    categories = ['Kadın', 'Erkek', 'Çocuk', 'Bebek', 'Home']
    cat_encoder = LabelEncoder()
    cat_encoder.fit(categories)
    
    past_categories = []
    next_category = []
    
    for i in range(n):
        if age[i] < 30:
            probs = [0.35, 0.40, 0.15, 0.05, 0.05]
        elif age[i] > 50:
            probs = [0.10, 0.15, 0.20, 0.25, 0.30]
        else:
            probs = [0.25, 0.30, 0.20, 0.10, 0.15]
        
        if discount_sensitivity[i] > 0.7:
            probs = [0.15, 0.20, 0.25, 0.25, 0.15]
        
        past = np.random.choice(categories, size=5, p=probs).tolist()
        past_categories.append(past)
        next_cat = np.random.choice(categories, p=probs)
        next_category.append(next_cat)
    
    # ----- YENİ: Recency etkisi U-şekilli (optimum 20 gün) -----
    # recency_effect = -0.0025*(recency - 20)^2 + 0.9  (maks 0.9, min 0.2)
    recency_effect = -0.0025 * (recency_days - 20)**2 + 0.9
    recency_effect = np.clip(recency_effect, 0.2, 0.9)
    
    # Hedef değişken: next_purchase_30d (U-şekilli recency kullan)
    log_odds = (recency_effect + 0.1*frequency + 0.005*(monetary_total/1000) +
                0.08*last_30_days_visits + 0.15*wishlist_count + 0.01*avg_discount_used -0.005*age - 2.0)
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
        'preferred_month': preferred_month,
        'age': age,
        'avg_discount_used': avg_discount_used,
        'next_purchase_30d': next_purchase_30d,
        'discount_sensitivity': discount_sensitivity,
        'past_categories': past_categories,
        'next_category': next_category
    })
    return df, cat_encoder, categories

df, cat_encoder, categories = generate_data()

# ------------------------------
# 2. ÖZELLİK MÜHENDİSLİĞİ: Geçmiş kategorileri sayısallaştır
# ------------------------------
def extract_category_features(row):
    past = row['past_categories']
    last_cat = past[-1] if len(past) > 0 else 'Erkek'
    cat_counts = {cat: past.count(cat) for cat in categories}
    return pd.Series([cat_counts.get(c,0) for c in categories] + [last_cat])

cat_feature_df = df.apply(extract_category_features, axis=1)
cat_feature_df.columns = [f'past_{c}_count' for c in categories] + ['last_category']
last_cat_onehot = pd.get_dummies(cat_feature_df['last_category'], prefix='last_cat')
cat_feature_df = pd.concat([cat_feature_df.drop('last_category', axis=1), last_cat_onehot], axis=1)

y_category = cat_encoder.transform(df['next_category'])
other_features = ['recency_days', 'frequency', 'monetary_total', 'last_30_days_visits',
                  'wishlist_count', 'preferred_channel', 'preferred_month', 'age', 'avg_discount_used']
X_cat = pd.concat([df[other_features], cat_feature_df], axis=1)

# Model eğitimi (kategori tahmini)
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_cat, y_category, test_size=0.2, random_state=42, stratify=y_category)
rf_cat = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
rf_cat.fit(X_train_cat, y_train_cat)
cat_accuracy = rf_cat.score(X_test_cat, y_test_cat)
rf_cat.fit(X_cat, y_category)
df['predicted_category'] = cat_encoder.inverse_transform(rf_cat.predict(X_cat))
df['predicted_category_proba'] = rf_cat.predict_proba(X_cat).max(axis=1)

# ------------------------------
# 3. MÜŞTERİ SEGMENTASYONU (K-Means)
# ------------------------------
features_seg = ['recency_days', 'frequency', 'monetary_total', 'last_30_days_visits', 'wishlist_count']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features_seg])

# Elbow grafiği (2D)
inertia = [KMeans(k, random_state=42, n_init=10).fit(X_scaled).inertia_ for k in range(2,8)]
fig_elbow, ax = plt.subplots(figsize=(6,4))
ax.plot(range(2,8), inertia, marker='o')
ax.set_title('Elbow Yöntemi (Optimum Küme Sayısı)')
ax.set_xlabel('Küme Sayısı')
ax.set_ylabel('Inertia')
st.pyplot(fig_elbow, use_container_width=False)

# 4 küme ile segmentasyon
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['segment'] = kmeans.fit_predict(X_scaled)
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

# PCA 3 boyut için (3 bileşen)
pca_3d = PCA(n_components=3)
pca_result_3d = pca_3d.fit_transform(X_scaled)
df['pca1'] = pca_result_3d[:,0]
df['pca2'] = pca_result_3d[:,1]
df['pca3'] = pca_result_3d[:,2]

# 3B interaktif grafik (Plotly)
fig_3d = px.scatter_3d(df, x='pca1', y='pca2', z='pca3',
                       color='segment_name', title='Segmentler (3B PCA Projeksiyonu)',
                       color_discrete_sequence=px.colors.qualitative.Set2,
                       hover_data=['customer_id', 'monetary_total', 'recency_days'])
fig_3d.update_layout(width=800, height=600)
st.plotly_chart(fig_3d, use_container_width=True)

# ------------------------------
# 4. NEXT PURCHASE MODELİ (Random Forest)
# ------------------------------
feature_cols = ['recency_days', 'frequency', 'monetary_total', 'last_30_days_visits',
                'wishlist_count', 'preferred_channel', 'preferred_month', 'age', 'avg_discount_used']
X_model = df[feature_cols]
y_model = df['next_purchase_30d']

rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X_model, y_model, scoring='roc_auc', cv=cv)
auc_mean = cv_scores.mean()

rf.fit(X_model, y_model)
df['next_purchase_prob'] = rf.predict_proba(X_model)[:,1]

# Özellik önemleri
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
fig_imp, ax = plt.subplots(figsize=(6,4))
importances.plot(kind='bar', ax=ax, width=0.7)
ax.set_title('Özellik Önemleri (Next Purchase)')
ax.set_ylabel('Önem')
plt.xticks(rotation=45, ha='right')
st.pyplot(fig_imp, use_container_width=False)

# ------------------------------
# 5. KİŞİSELLEŞTİRİLMİŞ ÖNERİLER (Müşteri Seçimi) - MEVSİME DUYARLI
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
    st.metric("Yaş", cust['age'])
    past_str = ", ".join(cust['past_categories'][-3:])
    st.metric("Son 3 Alışveriş Kategorisi", past_str)

with col2:
    st.subheader("🎯 Kişisel Öneriler")
    
    predicted_cat = cust['predicted_category']
    pred_proba = cust['predicted_category_proba']
    st.info(f"📂 **Tahmin Edilen Bir Sonraki Kategori:** {predicted_cat} (Güven: %{pred_proba*100:.0f})")
    
    # MEVSİMSEL ÜRÜN KATALOĞU
    month_season = {
        1: 'Kış', 2: 'Kış', 3: 'İlkbahar',
        4: 'İlkbahar', 5: 'İlkbahar', 6: 'Yaz',
        7: 'Yaz', 8: 'Yaz', 9: 'Sonbahar',
        10: 'Sonbahar', 11: 'Sonbahar', 12: 'Kış'
    }
    current_month = cust['preferred_month']
    season = month_season[current_month]
    
    seasonal_products = {
        'Kadın': {
            'Kış': ['Kadın Triko Kazak', 'Kadın Yün Mont', 'Kadın Kalın Taban Bot'],
            'İlkbahar': ['Kadın Trençkot', 'Kadın İnce Kazak', 'Kadın Babet Ayakkabı'],
            'Yaz': ['Kadın Elbise', 'Kadın Polo Tişört', 'Kadın Terlik'],
            'Sonbahar': ['Kadın Hırka', 'Kadın Yağmurluk', 'Kadın Çizme']
        },
        'Erkek': {
            'Kış': ['Erkek Yün Kazak', 'Erkek Pike Mont', 'Erkek Bot'],
            'İlkbahar': ['Erkek Trençkot', 'Erkek Oxford Gömlek', 'Erkek Spor Ayakkabı'],
            'Yaz': ['Erkek Polo Tişört', 'Erkek Şort', 'Erkek Terlik'],
            'Sonbahar': ['Erkek Hırka', 'Erkek Bomber Ceket', 'Erkek Günlük Ayakkabı']
        },
        'Çocuk': {
            'Kış': ['Çocuk Yün Kazak', 'Çocuk Mont', 'Çocuk Bot'],
            'İlkbahar': ['Çocuk Eşofman Takımı', 'Çocuk Gömlek', 'Çocuk Spor Ayakkabı'],
            'Yaz': ['Çocuk Polo Tişört', 'Çocuk Şort', 'Çocuk Terlik'],
            'Sonbahar': ['Çocuk Hırka', 'Çocuk Sweatshirt', 'Çocuk Günlük Ayakkabı']
        },
        'Bebek': {
            'Kış': ['Bebek Tulum (kalın)', 'Bebek Battaniye', 'Bebek Beresi'],
            'İlkbahar': ['Bebek Pamuklu Tulum', 'Bebek Hırka', 'Bebek Patik'],
            'Yaz': ['Bebek Kısa Kollu Tulum', 'Bebek Şapka', 'Bebek Sandalet'],
            'Sonbahar': ['Bebek Polar Tulum', 'Bebek Eldiven', 'Bebek Çorap']
        },
        'Home': {
            'Kış': ['Kalın Nevresim Takımı', 'Flanel Yatak Örtüsü', 'Yumuşak Halı'],
            'İlkbahar': ['Pamuklu Nevresim', 'Hafif Banyo Havlusu', 'Dekoratif Vazo'],
            'Yaz': ['İnce Nevresim', 'Plaj Havlusu', 'Fotoselli Mum'],
            'Sonbahar': ['Kadife Yastık Kılıfı', 'Pike Takımı', 'Dekoratif Sonbahar Çelengi']
        }
    }
    
    cat_season_products = seasonal_products.get(predicted_cat, seasonal_products['Erkek'])
    suitable_products = cat_season_products.get(season, cat_season_products['Kış'])
    recommended_product = suitable_products[0]
    
    st.success(f"📦 **Önerilen Ürün:** {recommended_product} ({season} mevsimine uygun)")
    st.caption("Öneri, tahmin edilen kategori ve mevsimsel faktöre göre yapılmıştır.")
    
    channel_map = {0: 'Web sitesi', 1: 'Mobil App', 2: 'Mağaza'}
    st.success(f"📱 **Önerilen İletişim Kanalı:** {channel_map[cust['preferred_channel']]}")
    
    month_names = ['Ocak', 'Şubat', 'Mart', 'Nisan', 'Mayıs', 'Haziran',
                   'Temmuz', 'Ağustos', 'Eylül', 'Ekim', 'Kasım', 'Aralık']
    best_month = month_names[cust['preferred_month'] - 1]
    st.warning(f"⏰ **En Uygun Zaman (Ay):** {best_month}")
    
    if cust['discount_sensitivity'] > 0.7:
        discount_str = "Yüksek indirim (%20 kupon) gönder"
    elif cust['discount_sensitivity'] > 0.4:
        discount_str = "Sepette %10 indirim teklifi"
    else:
        discount_str = "Özel koleksiyon tanıtımı, indirim yok"
    st.markdown(f"🏷️ **İndirim Stratejisi:** {discount_str}")

# ------------------------------
# 6. TOPLU PERFORMANS ÖZETİ
# ------------------------------
st.markdown("---")
st.header("📊 Model Performans Özeti")
col_met1, col_met2 = st.columns(2)
with col_met1:
    st.metric("Cross-Validation ROC AUC (Next Purchase)", f"{auc_mean:.3f}")
    st.caption("0.70+ iyi, 0.80+ mükemmel")
with col_met2:
    st.metric("Next Category Tahmin Doğruluğu (Accuracy)", f"{cat_accuracy:.2%}")
    st.caption("Test seti üzerinden hesaplanmıştır.")

# Segment dağılımı (2D bar)
fig_seg, ax = plt.subplots(figsize=(6,4))
df['segment_name'].value_counts().plot(kind='bar', ax=ax, color='lightblue', width=0.7)
ax.set_title('Segment Dağılımı')
ax.set_xlabel('Segment')
ax.set_ylabel('Müşteri Sayısı')
plt.xticks(rotation=45, ha='right')
st.pyplot(fig_seg, use_container_width=False)

st.success("✅ Prototip, veriye dayalı 'Bir Sonraki Kategori' tahmini ve mevsim duyarlı ürün önerisi yapmaktadır.")
