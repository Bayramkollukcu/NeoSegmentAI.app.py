import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Müşteri 360° - Segmentasyon & Tahmin", layout="wide")
st.title("🛍️ E-ticaret Müşteri Analitik Prototipi")
st.markdown("Ralph Lauren 4D Modeli esinlidir.")

# ------------------------------
# 1. VERİ OLUŞTURMA (YAŞ SÜTUNU YOK)
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
    month_probs = [0.06,0.06,0.07,0.08,0.09,0.12,0.13,0.12,0.09,0.07,0.06,0.05]
    preferred_month = np.random.choice(range(1,13), size=n, p=month_probs)
    discount_sensitivity = np.random.beta(2,5, size=n)
    avg_discount_used = (discount_sensitivity*50).astype(int).clip(0,50)

    # Gerçek yaş grubu (ground truth, sadece eğitim için)
    age_true = np.random.normal(35,10, size=n).astype(int).clip(18,70)
    def age_to_group(a):
        if a <= 25: return '18-25'
        elif a <= 35: return '26-35'
        elif a <= 45: return '36-45'
        else: return '46+'
    age_group_true = [age_to_group(a) for a in age_true]

    categories = ['Kadın', 'Erkek', 'Çocuk', 'Bebek', 'Home']
    cat_encoder = LabelEncoder()
    cat_encoder.fit(categories)
    
    past_categories = []
    next_category = []
    for i in range(n):
        if age_true[i] < 30:
            probs = [0.35, 0.40, 0.15, 0.05, 0.05]
        elif age_true[i] > 50:
            probs = [0.10, 0.15, 0.20, 0.25, 0.30]
        else:
            probs = [0.25, 0.30, 0.20, 0.10, 0.15]
        if discount_sensitivity[i] > 0.7:
            probs = [0.15, 0.20, 0.25, 0.25, 0.15]
        probs = np.array(probs) / np.sum(probs)
        past = np.random.choice(categories, size=5, p=probs).tolist()
        past_categories.append(past)
        next_cat = np.random.choice(categories, p=probs)
        next_category.append(next_cat)
    
    recency_effect = -0.0025 * (recency_days - 20)**2 + 0.9
    recency_effect = np.clip(recency_effect, 0.2, 0.9)
    log_odds = (recency_effect + 0.1*frequency + 0.005*(monetary_total/1000) +
                0.08*last_30_days_visits + 0.15*wishlist_count + 0.01*avg_discount_used - 2.0)
    prob_next = 1/(1+np.exp(-log_odds))
    next_purchase_30d = (np.random.rand(n) < prob_next).astype(int)

    marital_status_true = []
    has_children_true = []
    for a in age_true:
        if a < 30:
            p_bekar, p_evli = 0.75, 0.25
            child_base = 0.10
        elif a < 45:
            p_bekar, p_evli = 0.30, 0.70
            child_base = 0.60
        else:
            p_bekar, p_evli = 0.20, 0.80
            child_base = 0.40
        marital = np.random.choice(['Bekar','Evli'], p=[p_bekar, p_evli])
        marital_status_true.append(marital)
        if marital == 'Evli':
            child_prob = min(child_base + 0.20, 0.95)
        else:
            child_prob = child_base * 0.3
        child_prob = np.clip(child_prob + np.random.normal(0,0.05), 0.05, 0.95)
        has_children_true.append(np.random.choice([0,1], p=[1-child_prob, child_prob]))
    
    df = pd.DataFrame({
        'customer_id': customer_id,
        'recency_days': recency_days,
        'frequency': frequency,
        'monetary_total': monetary_total,
        'last_30_days_visits': last_30_days_visits,
        'wishlist_count': wishlist_count,
        'preferred_channel': preferred_channel,
        'preferred_month': preferred_month,
        'avg_discount_used': avg_discount_used,
        'next_purchase_30d': next_purchase_30d,
        'discount_sensitivity': discount_sensitivity,
        'past_categories': past_categories,
        'next_category': next_category
    })
    ground_truth = {
        'age_group': age_group_true,
        'marital_status': marital_status_true,
        'has_children': has_children_true
    }
    return df, cat_encoder, categories, ground_truth

df, cat_encoder, categories, ground_truth = generate_data()
n = len(df)

# ------------------------------
# 2. ÖZELLİK MÜHENDİSLİĞİ
# ------------------------------
def extract_last3_cat_features(row):
    past = row['past_categories']
    last3 = past[-3:] if len(past)>=3 else past + ['']*(3-len(past))
    counts = {cat: last3.count(cat) for cat in categories}
    return pd.Series([counts[c] for c in categories])

last3_cat_df = df.apply(extract_last3_cat_features, axis=1)
last3_cat_df.columns = [f'last3_{c}' for c in categories]

other_features = ['recency_days', 'frequency', 'monetary_total', 'last_30_days_visits',
                  'wishlist_count', 'preferred_channel', 'preferred_month', 'avg_discount_used']
X_base = df[other_features].copy()
X = pd.concat([X_base, last3_cat_df], axis=1)

# Yaş grubu tahmini (Random Forest)
age_groups = ['18-25', '26-35', '36-45', '46+']
age_encoder = LabelEncoder()
age_encoder.fit(age_groups)
y_age = age_encoder.transform(ground_truth['age_group'])
X_train_age, X_test_age, y_train_age, y_test_age = train_test_split(X, y_age, test_size=0.2, random_state=42, stratify=y_age)
rf_age = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
rf_age.fit(X_train_age, y_train_age)
age_acc = rf_age.score(X_test_age, y_test_age)
df['pred_age_group'] = age_encoder.inverse_transform(rf_age.predict(X))

# Yaş grubu dummy değişkenleri
age_dummies = pd.get_dummies(df['pred_age_group'], prefix='age')
df = pd.concat([df, age_dummies], axis=1)

# ------------------------------
# 3. MEDENİ DURUM VE ÇOCUK SAHİBİ OLMA MODELLERİ (Lojistik Regresyon)
# ------------------------------
X_marital = pd.concat([X, age_dummies], axis=1)
y_marital = np.array([1 if m == 'Evli' else 0 for m in ground_truth['marital_status']])
y_child = np.array(ground_truth['has_children'])

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_marital, y_marital, test_size=0.2, random_state=42, stratify=y_marital)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_marital, y_child, test_size=0.2, random_state=42, stratify=y_child)

scaler_marital = StandardScaler()
X_train_m_scaled = scaler_marital.fit_transform(X_train_m)
X_test_m_scaled = scaler_marital.transform(X_test_m)
model_marital = LogisticRegression(random_state=42)
model_marital.fit(X_train_m_scaled, y_train_m)
marital_acc = model_marital.score(X_test_m_scaled, y_test_m)

scaler_child = StandardScaler()
X_train_c_scaled = scaler_child.fit_transform(X_train_c)
X_test_c_scaled = scaler_child.transform(X_test_c)
model_child = LogisticRegression(random_state=42)
model_child.fit(X_train_c_scaled, y_train_c)
child_acc = model_child.score(X_test_c_scaled, y_test_c)

X_marital_scaled_all = scaler_marital.transform(X_marital)
X_child_scaled_all = scaler_child.transform(X_marital)
df['prob_married'] = model_marital.predict_proba(X_marital_scaled_all)[:, 1]
df['prob_child'] = model_child.predict_proba(X_child_scaled_all)[:, 1]

# ------------------------------
# 4. KATEGORİ GEÇİŞ MATRİSİ
# ------------------------------
all_transitions = []
for past in df['past_categories']:
    for i in range(len(past)-1):
        all_transitions.append((past[i], past[i+1]))
transition_df = pd.DataFrame(all_transitions, columns=['from', 'to'])
transition_matrix = pd.crosstab(transition_df['from'], transition_df['to'], normalize='index')

# ------------------------------
# 5. NEXT PURCHASE MODELİ (Random Forest) - DÜZELTİLMİŞ
# ------------------------------
# Dinamik olarak age_ ile başlayan sütunları al (df'de mevcut)
age_cols = [col for col in df.columns if col.startswith('age_')]
# Kullanılacak tüm sütunlar
feature_cols_np = other_features + [f'last3_{c}' for c in categories] + age_cols
# Sadece df'de bulunanları filtrele (ekstra güvenlik)
feature_cols_np = [col for col in feature_cols_np if col in df.columns]

X_np = df[feature_cols_np]
y_np = df['next_purchase_30d']
rf_np = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_np, X_np, y_np, scoring='roc_auc', cv=cv)
auc_mean = cv_scores.mean()
rf_np.fit(X_np, y_np)
df['next_purchase_prob'] = rf_np.predict_proba(X_np)[:,1]

# ------------------------------
# 6. MÜŞTERİ SEGMENTASYONU (K-Means)
# ------------------------------
features_seg = ['recency_days', 'frequency', 'monetary_total', 'last_30_days_visits', 'wishlist_count']
scaler_seg = StandardScaler()
X_scaled_seg = scaler_seg.fit_transform(df[features_seg])

# Elbow grafiği
inertia = [KMeans(k, random_state=42, n_init=10).fit(X_scaled_seg).inertia_ for k in range(2,8)]
fig_elbow, ax = plt.subplots(figsize=(6,4))
ax.plot(range(2,8), inertia, marker='o')
ax.set_title('Elbow Yöntemi (Optimum Küme Sayısı)')
ax.set_xlabel('Küme Sayısı')
ax.set_ylabel('Inertia')
st.pyplot(fig_elbow, use_container_width=False)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['segment'] = kmeans.fit_predict(X_scaled_seg)
seg_summary = df.groupby('segment').agg({'monetary_total':'mean', 'recency_days':'mean', 'frequency':'mean'}).reset_index()
def assign_segment_name(row):
    if row['monetary_total'] > 5000 and row['recency_days'] < 30:
        return 'Premium Sadık'
    elif row['monetary_total'] > 2000 and row['recency_days'] < 60:
        return 'Aktif Orta Sınıf'
    elif row['monetary_total'] < 2000 and row['recency_days'] < 30:
        return 'Düşük Değerli Yeni'
    else:
        return 'Riskli / Uyuyan'
seg_summary['segment_name'] = seg_summary.apply(assign_segment_name, axis=1)
seg_name_map = dict(zip(seg_summary['segment'], seg_summary['segment_name']))
df['segment_name'] = df['segment'].map(seg_name_map)

# PCA 3B görselleştirme
pca_3d = PCA(n_components=3)
pca_result_3d = pca_3d.fit_transform(X_scaled_seg)
df['pca1'], df['pca2'], df['pca3'] = pca_result_3d[:,0], pca_result_3d[:,1], pca_result_3d[:,2]
fig_3d = px.scatter_3d(df, x='pca1', y='pca2', z='pca3', color='segment_name',
                       title='Segmentler (3B PCA Projeksiyonu)',
                       color_discrete_sequence=px.colors.qualitative.Set2,
                       hover_data=['customer_id', 'monetary_total', 'recency_days'])
fig_3d.update_layout(width=800, height=600)
st.plotly_chart(fig_3d, use_container_width=True)

# ------------------------------
# 7. NEXT CATEGORY MODELİ
# ------------------------------
X_cat = pd.concat([X, age_dummies], axis=1)
y_cat = cat_encoder.transform(df['next_category'])
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_cat, y_cat, test_size=0.2, random_state=42, stratify=y_cat)
rf_cat = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
rf_cat.fit(X_train_cat, y_train_cat)
cat_accuracy = rf_cat.score(X_test_cat, y_test_cat)
rf_cat.fit(X_cat, y_cat)
df['predicted_category'] = cat_encoder.inverse_transform(rf_cat.predict(X_cat))
df['predicted_category_proba'] = rf_cat.predict_proba(X_cat).max(axis=1)

# ------------------------------
# 8. STREAMLIT ARABİRİMİ
# ------------------------------
st.markdown("---")
st.header("🔍 Müşteri Bazlı Analiz ve Öneriler")

customer_ids = df['customer_id'].tolist()
selected_id = st.selectbox("Bir müşteri ID seçin:", customer_ids)
cust = df[df['customer_id'] == selected_id].iloc[0]
idx = cust.name

col1, col2 = st.columns(2)
with col1:
    st.subheader("📋 Müşteri Profili")
    st.metric("Segment", cust['segment_name'])
    st.metric("Next Purchase Olasılığı", f"{cust['next_purchase_prob']:.0%}")
    st.metric("Son Alışveriş (gün)", f"{cust['recency_days']} gün")
    st.metric("Toplam Harcama (TL)", f"{cust['monetary_total']:,.0f}")
    st.metric("Wishlist Adedi", cust['wishlist_count'])
    st.metric("İndirim Duyarlılığı", f"{cust['discount_sensitivity']:.0%}")
    st.metric("Tahmini Yaş Aralığı", cust['pred_age_group'])
    st.metric("Evli Olma Olasılığı (Tahmini)", f"{cust['prob_married']:.0%}")
    st.metric("Çocuk Sahibi Olma Olasılığı (Tahmini)", f"{cust['prob_child']:.0%}")
    st.caption(f"Not: Gerçek yaş grubu: {ground_truth['age_group'][idx]}, "
               f"Gerçek medeni durum: {ground_truth['marital_status'][idx]}, "
               f"Gerçek çocuk: {'Evet' if ground_truth['has_children'][idx] else 'Hayır'}")

with col2:
    st.subheader("🎯 Kişisel Öneriler")
    predicted_cat = cust['predicted_category']
    pred_proba = cust['predicted_category_proba']
    st.info(f"📂 **Tahmin Edilen Bir Sonraki Kategori:** {predicted_cat} (Güven: %{pred_proba*100:.0f})")
    
    last_cat = cust['past_categories'][-1]
    if last_cat in transition_matrix.index:
        trans_probs = transition_matrix.loc[last_cat].sort_values(ascending=False)
        st.markdown("🔄 **Kategori Geçiş Olasılıkları (son kategori bazlı):**")
        for cat, prob in trans_probs.head(3).items():
            st.write(f"- {last_cat} → {cat}: %{prob*100:.0f}")
    else:
        st.write("Geçiş bilgisi yetersiz.")
    
    month_season = {1:'Kış',2:'Kış',3:'İlkbahar',4:'İlkbahar',5:'İlkbahar',6:'Yaz',
                    7:'Yaz',8:'Yaz',9:'Sonbahar',10:'Sonbahar',11:'Sonbahar',12:'Kış'}
    season = month_season[cust['preferred_month']]
    seasonal_products = {
        'Kadın': {'Kış':['Kadın Triko Kazak'], 'İlkbahar':['Kadın Trençkot'], 'Yaz':['Kadın Elbise'], 'Sonbahar':['Kadın Hırka']},
        'Erkek': {'Kış':['Erkek Yün Kazak'], 'İlkbahar':['Erkek Trençkot'], 'Yaz':['Erkek Polo Tişört'], 'Sonbahar':['Erkek Hırka']},
        'Çocuk': {'Kış':['Çocuk Yün Kazak'], 'İlkbahar':['Çocuk Eşofman'], 'Yaz':['Çocuk Polo'], 'Sonbahar':['Çocuk Sweatshirt']},
        'Bebek': {'Kış':['Bebek Tulum'], 'İlkbahar':['Bebek Pamuklu'], 'Yaz':['Bebek Kısa Kollu'], 'Sonbahar':['Bebek Polar']},
        'Home': {'Kış':['Kalın Nevresim'], 'İlkbahar':['Pamuklu Nevresim'], 'Yaz':['İnce Nevresim'], 'Sonbahar':['Kadife Yastık']}
    }
    rec_product = seasonal_products.get(predicted_cat, seasonal_products['Erkek']).get(season, ['Ürün'])[0]
    st.success(f"📦 **Önerilen Ürün:** {rec_product} ({season} mevsimine uygun)")
    
    channel_map = {0:'Web sitesi',1:'Mobil App',2:'Mağaza'}
    st.success(f"📱 **Önerilen İletişim Kanalı:** {channel_map[cust['preferred_channel']]}")
    
    month_names = ['Ocak','Şubat','Mart','Nisan','Mayıs','Haziran','Temmuz','Ağustos','Eylül','Ekim','Kasım','Aralık']
    best_month = month_names[cust['preferred_month']-1]
    st.warning(f"⏰ **En Uygun Zaman (Ay):** {best_month}")
    
    if cust['discount_sensitivity'] > 0.7:
        disc_str = "Yüksek indirim (%20 kupon) gönder"
    elif cust['discount_sensitivity'] > 0.4:
        disc_str = "Sepette %10 indirim teklifi"
    else:
        disc_str = "Özel koleksiyon tanıtımı, indirim yok"
    st.markdown(f"🏷️ **İndirim Stratejisi:** {disc_str}")

# ------------------------------
# 9. PERFORMANS ÖZETİ
# ------------------------------
st.markdown("---")
st.header("📊 Model Performans Özeti")
col_met1, col_met2, col_met3 = st.columns(3)
with col_met1:
    st.metric("Yaş Aralığı Tahmin Doğruluğu", f"{age_acc:.2%}")
with col_met2:
    st.metric("Evli Olma Tahmin Doğruluğu", f"{marital_acc:.2%}")
with col_met3:
    st.metric("Çocuk Sahibi Olma Doğruluğu", f"{child_acc:.2%}")
st.metric("Next Purchase ROC AUC (CV)", f"{auc_mean:.3f}")
st.metric("Next Category Doğruluğu", f"{cat_accuracy:.2%}")

st.success("✅ Prototip başarıyla çalışıyor. Yaş aralığı, evlilik ve çocuk sahibi olma olasılıkları tahmin edilmektedir.")
