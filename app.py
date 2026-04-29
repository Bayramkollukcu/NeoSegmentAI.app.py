import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Müşteri 360° Analitik", layout="wide")
st.title("🛍️ E-ticaret Müşteri Analitik Prototipi")
st.markdown("Ralph Lauren 4D Modeli esinlidir. Gerçek coğrafi veriler (bölge/şehir/ilçe) kullanılır. İndirim duyarlılığı, kullanılan ortalama indirimden türetilmiştir.")

# ------------------------------
# 0. TÜRKİYE BÖLGE, ŞEHİR VE İLÇE TANIMLARI
# ------------------------------
region_cities = {
    'Marmara': ['İstanbul', 'Bursa', 'Kocaeli'],
    'Ege': ['İzmir', 'Manisa', 'Aydın'],
    'Akdeniz': ['Antalya', 'Adana', 'Mersin'],
    'İç Anadolu': ['Ankara', 'Konya', 'Kayseri'],
    'Karadeniz': ['Samsun', 'Trabzon', 'Ordu'],
    'Doğu Anadolu': ['Erzurum', 'Van', 'Malatya'],
    'Güneydoğu Anadolu': ['Gaziantep', 'Diyarbakır', 'Şanlıurfa']
}
city_districts = {
    'İstanbul': ['Kadıköy', 'Beşiktaş', 'Şişli', 'Ümraniye', 'Bakırköy'],
    'Bursa': ['Osmangazi', 'Nilüfer', 'Yıldırım', 'Mudanya'],
    'Kocaeli': ['İzmit', 'Gebze', 'Körfez', 'Darıca'],
    'İzmir': ['Konak', 'Karşıyaka', 'Bornova', 'Buca', 'Çeşme'],
    'Manisa': ['Yunusemre', 'Akhisar', 'Turgutlu', 'Salihli'],
    'Aydın': ['Efeler', 'Kuşadası', 'Didim', 'Nazilli'],
    'Antalya': ['Muratpaşa', 'Kepez', 'Alanya', 'Manavgat', 'Konyaaltı'],
    'Adana': ['Seyhan', 'Yüreğir', 'Çukurova', 'Sarıçam'],
    'Mersin': ['Akdeniz', 'Yenişehir', 'Tarsus', 'Erdemli'],
    'Ankara': ['Çankaya', 'Keçiören', 'Etimesgut', 'Yenimahalle', 'Mamak'],
    'Konya': ['Selçuklu', 'Meram', 'Karatay', 'Ereğli'],
    'Kayseri': ['Melikgazi', 'Kocasinan', 'Talas', 'Develi'],
    'Samsun': ['İlkadım', 'Atakum', 'Canik', 'Bafra'],
    'Trabzon': ['Ortahisar', 'Akçaabat', 'Yomra', 'Araklı'],
    'Ordu': ['Altınordu', 'Ünye', 'Fatsa', 'Perşembe'],
    'Erzurum': ['Yakutiye', 'Palandöken', 'Aziziye', 'Oltu'],
    'Van': ['İpekyolu', 'Tuşba', 'Edremit', 'Erciş'],
    'Malatya': ['Battalgazi', 'Yeşilyurt', 'Doğanşehir', 'Akçadağ'],
    'Gaziantep': ['Şahinbey', 'Şehitkamil', 'Oğuzeli', 'Nizip'],
    'Diyarbakır': ['Kayapınar', 'Bağlar', 'Yenişehir', 'Sur'],
    'Şanlıurfa': ['Eyyübiye', 'Haliliye', 'Karaköprü', 'Siverek']
}
all_cities = [city for cities in region_cities.values() for city in cities]
all_regions = list(region_cities.keys())

# ------------------------------
# 1. VERİ OLUŞTURMA (SENTETİK, COĞRAFİ + İNDİRİM DUYARLILIĞI DÜZELTİLDİ)
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
    
    # avg_discount_used rastgele üret, discount_sensitivity'i bundan türet
    avg_discount_used = np.random.exponential(scale=12, size=n).astype(int).clip(0,50)
    discount_sensitivity = (avg_discount_used / 50) + np.random.normal(0, 0.05, size=n)
    discount_sensitivity = np.clip(discount_sensitivity, 0, 1)

    # Ground truth (gerçek) cinsiyet, yaş vb.
    gender_true = np.random.choice(['Kadın', 'Erkek'], size=n, p=[0.5, 0.5])
    age_true = np.random.normal(35,10, size=n).astype(int).clip(18,70)
    def age_to_group(a):
        if a <= 25: return '18-25'
        elif a <= 35: return '26-35'
        elif a <= 45: return '36-45'
        else: return '46+'
    age_group_true = [age_to_group(a) for a in age_true]

    # Coğrafi bilgiler (bölge, şehir, ilçe)
    region_weights = [0.25, 0.15, 0.15, 0.2, 0.1, 0.05, 0.1]
    region_list = np.random.choice(all_regions, size=n, p=region_weights)
    city_list = []
    district_list = []
    for reg in region_list:
        city = np.random.choice(region_cities[reg])
        city_list.append(city)
        district = np.random.choice(city_districts.get(city, ['Merkez']))
        district_list.append(district)

    # Kategoriler (alışveriş kategorileri)
    categories = ['Kadın', 'Erkek', 'Çocuk', 'Bebek', 'Home']
    cat_encoder = LabelEncoder()
    cat_encoder.fit(categories)
    
    past_categories = []
    next_category = []
    for i in range(n):
        if gender_true[i] == 'Kadın':
            base_probs = [0.45, 0.25, 0.15, 0.05, 0.10]
        else:
            base_probs = [0.25, 0.45, 0.15, 0.05, 0.10]
        if age_true[i] < 30:
            age_probs = [0.35, 0.40, 0.15, 0.05, 0.05]
        elif age_true[i] > 50:
            age_probs = [0.10, 0.15, 0.20, 0.25, 0.30]
        else:
            age_probs = [0.25, 0.30, 0.20, 0.10, 0.15]
        geo_effect = [1.0, 1.0, 1.0, 1.0, 1.0]
        reg = region_list[i]
        if reg == 'Ege':
            geo_effect[0] = 1.3  # Kadın
        elif reg == 'İç Anadolu':
            geo_effect[4] = 1.4  # Home
        elif reg == 'Karadeniz':
            geo_effect[1] = 1.3  # Erkek
        probs = (np.array(base_probs) + np.array(age_probs)) / 2
        probs = probs * np.array(geo_effect)
        if discount_sensitivity[i] > 0.7:
            probs = [0.15, 0.20, 0.25, 0.25, 0.15]
        probs = np.array(probs) / np.sum(probs)
        past = np.random.choice(categories, size=5, p=probs).tolist()
        past_categories.append(past)
        next_cat = np.random.choice(categories, p=probs)
        next_category.append(next_cat)
    
    # Next purchase 30d (binary)
    recency_effect = -0.0025 * (recency_days - 20)**2 + 0.9
    recency_effect = np.clip(recency_effect, 0.2, 0.9)
    log_odds = (recency_effect + 0.1*frequency + 0.005*(monetary_total/1000) +
                0.08*last_30_days_visits + 0.15*wishlist_count + 0.01*avg_discount_used - 2.0)
    prob_next = 1/(1+np.exp(-log_odds))
    next_purchase_30d = (np.random.rand(n) < prob_next).astype(int)

    # Bekleme süresi (gün cinsinden ground truth)
    wait_days_true = np.random.poisson(lam=25, size=n) + (recency_days * 0.3).astype(int)
    wait_days_true = np.clip(wait_days_true, 1, 90)

    # Medeni durum ve çocuk sahibi olma (ground truth)
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
        'region': region_list,
        'city': city_list,
        'district': district_list,
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
        'next_category': next_category,
        'wait_days_true': wait_days_true
    })
    ground_truth = {
        'gender': gender_true,
        'age_group': age_group_true,
        'marital_status': marital_status_true,
        'has_children': has_children_true
    }
    return df, cat_encoder, categories, ground_truth

df, cat_encoder, categories, ground_truth = generate_data()
n = len(df)

# ------------------------------
# 2. ÖZELLİK MÜHENDİSLİĞİ (SON 3 KATEGORİ + COĞRAFİ ENCODE)
# ------------------------------
def extract_last3_cat_features(row):
    past = row['past_categories']
    last3 = past[-3:] if len(past)>=3 else past + ['TAMAM']*(3-len(past))
    counts = {cat: last3.count(cat) for cat in categories}
    return pd.Series([counts[c] for c in categories])

last3_cat_df = df.apply(extract_last3_cat_features, axis=1)
last3_cat_df.columns = [f'last3_{c}' for c in categories]

other_features = ['recency_days', 'frequency', 'monetary_total', 'last_30_days_visits',
                  'wishlist_count', 'preferred_channel', 'preferred_month', 'avg_discount_used']
X_base = df[other_features].copy()
X = pd.concat([X_base, last3_cat_df], axis=1)

# Coğrafi değişkenleri sayısal hale getir
city_encoder = LabelEncoder()
region_encoder = LabelEncoder()
district_encoder = LabelEncoder()
df['city_code'] = city_encoder.fit_transform(df['city'])
df['region_code'] = region_encoder.fit_transform(df['region'])
df['district_code'] = district_encoder.fit_transform(df['district'])

geo_features = ['city_code', 'region_code', 'district_code']
X_geo = df[geo_features].copy()
X = pd.concat([X, X_geo], axis=1)

# ------------------------------
# 3. YAŞ ARALIĞI TAHMİNİ (Random Forest)
# ------------------------------
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
# 4. CİNSİYET TAHMİNİ (Lojistik Regresyon)
# ------------------------------
X_gender = X.copy()
y_gender = (ground_truth['gender'] == 'Kadın').astype(int)
X_train_g, X_test_g, y_train_g, y_test_g = train_test_split(X_gender, y_gender, test_size=0.2, random_state=42, stratify=y_gender)
scaler_gender = StandardScaler()
X_train_g_scaled = scaler_gender.fit_transform(X_train_g)
X_test_g_scaled = scaler_gender.transform(X_test_g)
model_gender = LogisticRegression(random_state=42)
model_gender.fit(X_train_g_scaled, y_train_g)
gender_acc = model_gender.score(X_test_g_scaled, y_test_g)
X_gender_scaled_all = scaler_gender.transform(X_gender)
df['prob_female'] = model_gender.predict_proba(X_gender_scaled_all)[:, 1]

# ------------------------------
# 5. BEKLEME SÜRESİ (Random Forest Regresyon)
# ------------------------------
X_wait = X.copy()
y_wait = df['wait_days_true']
X_train_wait, X_test_wait, y_train_wait, y_test_wait = train_test_split(X_wait, y_wait, test_size=0.2, random_state=42)
rf_wait = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
rf_wait.fit(X_train_wait, y_train_wait)
wait_rmse = np.sqrt(np.mean((rf_wait.predict(X_test_wait) - y_test_wait)**2))
df['predicted_wait_days'] = rf_wait.predict(X_wait).astype(int)

# ------------------------------
# 6. MEDENİ DURUM VE ÇOCUK (Lojistik Regresyon)
# ------------------------------
X_marital = X.copy()
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
# 7. KATEGORİ GEÇİŞ MATRİSİ
# ------------------------------
all_transitions = []
for past in df['past_categories']:
    for i in range(len(past)-1):
        all_transitions.append((past[i], past[i+1]))
transition_df = pd.DataFrame(all_transitions, columns=['from', 'to'])
transition_matrix = pd.crosstab(transition_df['from'], transition_df['to'], normalize='index')

# ------------------------------
# 8. NEXT PURCHASE MODELİ (Random Forest, CV)
# ------------------------------
X_np = X.copy()
y_np = df['next_purchase_30d']
rf_np = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf_np, X_np, y_np, scoring='roc_auc', cv=cv)
auc_mean = cv_scores.mean()
rf_np.fit(X_np, y_np)
df['next_purchase_prob'] = rf_np.predict_proba(X_np)[:,1]

# ------------------------------
# ------------------------------
# 9. MÜŞTERİ SEGMENTASYONU (K-Means, davranışsal verilerle) - 4 net segment
# ------------------------------
features_seg = ['recency_days', 'frequency', 'monetary_total', 'last_30_days_visits', 'wishlist_count']
scaler_seg = StandardScaler()
X_scaled_seg = scaler_seg.fit_transform(df[features_seg])

# Elbow grafiği (küçük boyutta)
inertia = [KMeans(k, random_state=42, n_init=10).fit(X_scaled_seg).inertia_ for k in range(2,8)]
fig_elbow, ax = plt.subplots(figsize=(4,3))
ax.plot(range(2,8), inertia, marker='o')
ax.set_title('Elbow Yöntemi (Optimum Küme Sayısı)')
ax.set_xlabel('Küme Sayısı')
ax.set_ylabel('Inertia')
st.pyplot(fig_elbow, use_container_width=False)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df['segment'] = kmeans.fit_predict(X_scaled_seg)

# Kümelerin merkezlerini al (ölçeklendirilmemiş orijinal değerler üzerinden)
cluster_centers = df.groupby('segment')[features_seg].mean()

# Merkezleri monetary_total (büyükten küçüğe) ve recency_days (küçükten büyüğe) sırala
# Önce monetary_total yüksek olanlar daha değerli. Aynı monetary grubunda recency düşük olan daha aktif.
cluster_centers['score'] = cluster_centers['monetary_total'] / 1000 - cluster_centers['recency_days'] / 100
cluster_centers = cluster_centers.sort_values('score', ascending=False)

# İsimlendirme (en yüksek skordan en düşüğe)
names = ['Premium Sadık', 'Aktif Orta Sınıf', 'Düşük Değerli Yeni', 'Riskli / Uyuyan']
seg_name_map = {}
for i, seg in enumerate(cluster_centers.index):
    seg_name_map[seg] = names[i]

df['segment_name'] = df['segment'].map(seg_name_map)

# ------------------------------
# 10. NEXT CATEGORY MODELİ (Random Forest)
# ------------------------------
X_cat = X.copy()
y_cat = cat_encoder.transform(df['next_category'])
X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_cat, y_cat, test_size=0.2, random_state=42, stratify=y_cat)
rf_cat = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
rf_cat.fit(X_train_cat, y_train_cat)
cat_accuracy = rf_cat.score(X_test_cat, y_test_cat)
rf_cat.fit(X_cat, y_cat)
df['predicted_category'] = cat_encoder.inverse_transform(rf_cat.predict(X_cat))
df['predicted_category_proba'] = rf_cat.predict_proba(X_cat).max(axis=1)

# ------------------------------
# 11. STREAMLIT ARABİRİMİ (Müşteri seçimi, kompakt profil düzeni)
# ------------------------------
st.markdown("---")
st.header("🔍 Müşteri Bazlı Analiz ve Öneriler")
customer_ids = df['customer_id'].tolist()
selected_id = st.selectbox("Bir müşteri ID seçin:", customer_ids)
cust = df[df['customer_id'] == selected_id].iloc[0]
idx = cust.name

col1, col2 = st.columns([1.2, 1])
with col1:
    st.subheader("📋 Müşteri Profili")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Segment", cust['segment_name'])
        st.metric("Son Alışveriş", f"{cust['recency_days']} gün")
        st.metric("Toplam Harcama", f"₺{cust['monetary_total']:,.0f}")
        st.metric("İndirim Duyarlılığı", f"%{cust['discount_sensitivity']*100:.0f}")
        st.metric("Kadın Olma Olasılığı", f"%{cust['prob_female']*100:.0f}")
        st.metric("Evli Olma Olasılığı", f"%{cust['prob_married']*100:.0f}")
    with c2:
        st.metric("Next Purchase (30g)", f"%{cust['next_purchase_prob']*100:.0f}")
        st.metric("Bekleme Süresi (tahmini)", f"{cust['predicted_wait_days']} gün")
        st.metric("Wishlist Adedi", cust['wishlist_count'])
        st.metric("Yaş Aralığı (tahmini)", cust['pred_age_group'])
        st.metric("Erkek Olma Olasılığı", f"%{(1-cust['prob_female'])*100:.0f}")
        st.metric("Çocuk Sahibi Olma", f"%{cust['prob_child']*100:.0f}")
    with c3:
        st.metric("Bölge", cust['region'])
        st.metric("Şehir", cust['city'])
        st.metric("İlçe", cust['district'])
        st.markdown("---")
        st.caption(f"**Gerçek bilgiler (doğrulama)**  \n"
                   f"Cinsiyet: {ground_truth['gender'][idx]}  \n"
                   f"Yaş grubu: {ground_truth['age_group'][idx]}  \n"
                   f"Medeni durum: {ground_truth['marital_status'][idx]}  \n"
                   f"Çocuk: {'Evet' if ground_truth['has_children'][idx] else 'Hayır'}")

with col2:
    st.subheader("🎯 Kişisel Öneriler")
    predicted_cat = cust['predicted_category']
    pred_proba = cust['predicted_category_proba']
    st.info(f"📂 **Bir Sonraki Kategori Tahmini:** {predicted_cat} (Güven: %{pred_proba*100:.0f})")
    
    last_cat = cust['past_categories'][-1]
    if last_cat in transition_matrix.index:
        trans_probs = transition_matrix.loc[last_cat].sort_values(ascending=False)
        st.markdown("🔄 **Kategori Geçiş Olasılıkları (son kategoriye göre):**")
        for cat, prob in trans_probs.head(3).items():
            st.write(f"- {last_cat} → {cat}: %{prob*100:.0f}")
    else:
        st.write("Geçiş bilgisi yetersiz.")
    
    # Mevsimsel ürün önerisi
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
# 12. PERFORMANS ÖZETİ
# ------------------------------
st.markdown("---")
st.header("📊 Model Performans Özeti")
col_met1, col_met2, col_met3, col_met4, col_met5 = st.columns(5)
with col_met1:
    st.metric("Cinsiyet Doğruluğu", f"{gender_acc:.2%}")
with col_met2:
    st.metric("Yaş Aralığı Doğruluğu", f"{age_acc:.2%}")
with col_met3:
    st.metric("Evli Olma Doğruluğu", f"{marital_acc:.2%}")
with col_met4:
    st.metric("Çocuk Sahibi Doğruluğu", f"{child_acc:.2%}")
with col_met5:
    st.metric("Bekleme Süresi RMSE (gün)", f"{wait_rmse:.1f}")
st.metric("Next Purchase ROC AUC (CV)", f"{auc_mean:.3f}")
st.metric("Next Category Doğruluğu", f"{cat_accuracy:.2%}")

st.success("✅ Prototip, güncel indirim duyarlılığı türetimi ve coğrafi bilgilerle çalışmaktadır.")
