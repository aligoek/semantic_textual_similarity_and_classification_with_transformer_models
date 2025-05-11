import pandas as pd
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import random
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanilan cihaz: {device}")

def modelYukle(model_name):
    if "jina" in model_name.lower() or "e5" in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if "jina" in model_name.lower():
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float32).to(device)
        else:
            model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)

    if tokenizer.pad_token is None and hasattr(tokenizer, 'eos_token'):
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def calculate_embeddings(texts, model, tokenizer, max_length=512):
    embeddings = []
    batch_size = 1
    num_texts = len(texts)
    processed_count = 0
    last_percentage = -1

    print("Embeddings hesaplaniyor.") 

    for i in range(0, num_texts, batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = []

        for text in batch_texts:
            if not isinstance(text, str) or pd.isna(text) or text.strip() == '':
                if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
                    batch_embeddings.append(np.zeros(model.config.hidden_size))
                else:
                    batch_embeddings.append(np.zeros(768))
                continue

            try:
                inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(device)

                with torch.no_grad():
                    outputs = model(**inputs)

                if hasattr(outputs, 'last_hidden_state'):
                    embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
                else:
                    embedding = outputs[0][:, 0, :].squeeze().cpu().numpy()

                if embedding.ndim == 1:
                     expected_dim = model.config.hidden_size if hasattr(model, 'config') and hasattr(model.config, 'hidden_size') else 768
                     if embedding.shape[0] != expected_dim:
                         print(f"Uyari: Beklenmeyen embedding boyutu {embedding.shape[0]}. Sifir vektor kullaniliyor.")
                         embedding = np.zeros(expected_dim)
                else:
                     print(f"Uyari: Beklenmeyen embedding boyutu {embedding.shape}. Sifir vektor kullaniliyor.")
                     expected_dim = model.config.hidden_size if hasattr(model, 'config') and hasattr(model.config, 'hidden_size') else 768
                     embedding = np.zeros(expected_dim)


                batch_embeddings.append(embedding)
            except Exception as e:
                print(f"Metin islenirken hata: {str(e)[:100]}...")
                if hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
                    batch_embeddings.append(np.zeros(model.config.hidden_size))
                else:
                    batch_embeddings.append(np.zeros(768))

        embeddings.extend(batch_embeddings)
        processed_count += len(batch_texts) 

        current_percentage = int((processed_count / num_texts) * 100)
        if current_percentage >= last_percentage + 20:
            print(f"Embeddings hesaplaniyor: {current_percentage}%")
            last_percentage = current_percentage

    if last_percentage < 100:
         print(f"Embeddings hesaplaniyor: 100%")


    if embeddings:
        ref_dim = embeddings[0].shape[0]
        for emb in embeddings:
            if emb.shape[0] != ref_dim:
                 print(f"Uyari: Embedding boyutlarinda tutarsizlik tespit edildi.")
                 embeddings = [e if e.shape[0] == ref_dim else np.zeros(ref_dim) for e in embeddings]
                 break

    return np.array(embeddings)

def topAccuracyHesapla(question_embeddings, answer_embeddings, k=5):
    similarity_matrix = cosine_similarity(question_embeddings, answer_embeddings)

    top1_correct = 0
    topk_correct = 0

    for i in range(len(question_embeddings)):
        similarity_scores = similarity_matrix[i]
        most_similar_indices = similarity_scores.argsort()[::-1][:k]
        true_answer_index = i

        if most_similar_indices[0] == true_answer_index:
            top1_correct += 1

        if true_answer_index in most_similar_indices:
            topk_correct += 1

    num_questions = len(question_embeddings)
    top1_accuracy = top1_correct / num_questions if num_questions > 0 else 0
    topk_accuracy = topk_correct / num_questions if num_questions > 0 else 0

    return top1_accuracy, topk_accuracy

def sorudanCevaba(data, model_name, model, tokenizer):
    print(f"\n{model_name} modeli ile 'Sorudan cevaba cozumu basladi")

    sample_df = data.copy()
    sample_df.reset_index(drop=True, inplace=True)

    preference_column_name = "Hangisi iyi? (1: gpt4o daha iyi, 2: deepseek daha iyi, 3: ikisi de yeterince iyi, 4: ikisi de kötü)"

    required_cols = ['Soru', 'gpt4o cevabı', 'deepseek cevabı']

    question_embeddings = calculate_embeddings(sample_df['Soru'].tolist(), model, tokenizer)
    gpt4o_answer_embeddings = calculate_embeddings(sample_df['gpt4o cevabı'].tolist(), model, tokenizer)
    deepseek_answer_embeddings = calculate_embeddings(sample_df['deepseek cevabı'].tolist(), model, tokenizer)

    print("GPT4o icin Top-1 ve Top-5 basarilari hesaplaniyo")
    gpt4o_top1, gpt4o_top5 = topAccuracyHesapla(question_embeddings, gpt4o_answer_embeddings, k=5)
    print(f"GPT4o - Top1 Basarisi {gpt4o_top1:.4f}")
    print(f"GPT4o - Top5 Basarisi {gpt4o_top5:.4f}")

    print("Deepseek icin Top-1 ve Top-5 basarilari hesaplaniyo")
    deepseek_top1, deepseek_top5 = topAccuracyHesapla(question_embeddings, deepseek_answer_embeddings, k=5)
    print(f"Deepseek - Top1 Basarisi {deepseek_top1:.4f}")
    print(f"Deepseek - Top5 Basarisi {deepseek_top5:.4f}")

    preference_groups = {}
    if preference_column_name in sample_df.columns:
        preference_counts = sample_df[preference_column_name].value_counts()
        print("\nOrneklenmis verideki tercihlerin dagilimi")
        for pref, count in preference_counts.items():
            print(f"Tercih {pref}: {count} ornek ({count/len(sample_df)*100:.2f}%)")

        sample_df['Hangisi iyi?_numeric'] = pd.to_numeric(sample_df[preference_column_name], errors='coerce')
        valid_pref_df = sample_df.dropna(subset=['Hangisi iyi?_numeric']).copy()

        if not valid_pref_df.empty:
            valid_indices = valid_pref_df.index
            q_emb_valid = question_embeddings[valid_indices]
            gpt_emb_valid = gpt4o_answer_embeddings[valid_indices]
            ds_emb_valid = deepseek_answer_embeddings[valid_indices]

            valid_preferences = sorted(valid_pref_df['Hangisi iyi?_numeric'].unique())
            for pref in valid_preferences:
                mask = valid_pref_df['Hangisi iyi?_numeric'] == pref
                if mask.sum() == 0: continue

                q_emb_group = q_emb_valid[mask.values]
                gpt_emb_group = gpt_emb_valid[mask.values]
                ds_emb_group = ds_emb_valid[mask.values]

                gpt_sims_mean = np.mean([cosine_similarity([q], [g])[0][0] for q, g in zip(q_emb_group, gpt_emb_group)]) if len(q_emb_group) > 0 else 0
                ds_sims_mean = np.mean([cosine_similarity([q], [d])[0][0] for q, d in zip(q_emb_group, ds_emb_group)]) if len(q_emb_group) > 0 else 0

                preference_groups[pref] = {
                    'sayi': mask.sum(),
                    'ortalama_gpt4o_benzerligi': gpt_sims_mean,
                    'ortalama_deepseek_benzerligi': ds_sims_mean
                }

            print("\nTercih grubuna gore ortalama benzerlikler:")
            for pref, data in preference_groups.items():
                print(f"Tercih {pref} (n={data['sayi']}):")
                print(f"  Ortalama GPT4o benzerligi: {data['ortalama_gpt4o_benzerligi']:.4f}")
                print(f"  Ortalama Deepseek benzerligi: {data['ortalama_deepseek_benzerligi']:.4f}")

            plot_similarity_by_preference(sample_df, question_embeddings, gpt4o_answer_embeddings, deepseek_answer_embeddings, model_name, preference_column_name)
        else:
            print("Gecerli tercih verisi bulunamadi.")

    return {
        'model': model_name,
        'gpt4o_top1': gpt4o_top1,
        'gpt4o_top5': gpt4o_top5,
        'deepseek_top1': deepseek_top1,
        'deepseek_top5': deepseek_top5,
        'tercih_gruplari': preference_groups
    }

def plot_similarity_by_preference(df, q_emb, gpt_emb, ds_emb, model_name, preference_column_name):

    gpt_sims = [cosine_similarity([q], [g])[0][0] for q, g in zip(q_emb, gpt_emb)]
    ds_sims = [cosine_similarity([q], [d])[0][0] for q, d in zip(q_emb, ds_emb)]
    sim_diff = [g - d for g, d in zip(gpt_sims, ds_sims)]

    plot_df = pd.DataFrame({
        'Tercih': df[preference_column_name],
        'GPT4o_Benzerligi': gpt_sims,
        'Deepseek_Benzerligi': ds_sims,
        'Benzerlik_Farki': sim_diff
    })

    plot_df = plot_df.dropna(subset=['Tercih'])

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Tercih', y='Benzerlik_Farki', data=plot_df)
    plt.title(f'Tercihe Gore GPT4o - Deepseek Benzerlik Farki ({model_name})')
    plt.xlabel('Tercih (1:GPT4o daha iyi, 2:Deepseek daha iyi, 3:Ikisi de iyi, 4:Ikisi de kotu)')
    plt.ylabel('GPT4o Benzerligi - Deepseek Benzerligi')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig(f'{model_name}_benzerlik_farki_tercihe_gore.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    plot_df_melted = pd.melt(plot_df,
                                id_vars=['Tercih'],
                                value_vars=['GPT4o_Benzerligi', 'Deepseek_Benzerligi'],
                                var_name='Model', value_name='Benzerlik')

    sns.boxplot(x='Tercih', y='Benzerlik', hue='Model', data=plot_df_melted)
    plt.title(f'Tercihe ve Modele Gore Soru-Cevap Benzerligi ({model_name})')
    plt.xlabel('Tercih (1:GPT4o daha iyi, 2:Deepseek daha iyi, 3:Ikisi de iyi, 4:Ikisi de kotu)')
    plt.ylabel('Kosinus Benzerligi')
    plt.tight_layout()
    plt.savefig(f'{model_name}_benzerlik_tercihe_gore.png')
    plt.close()

def create_feature_vectors(q_emb, gpt_emb, ds_emb):
    feature_vectors = {}

    feature_vectors['s'] = q_emb
    feature_vectors['g'] = gpt_emb
    feature_vectors['d'] = ds_emb

    feature_vectors['s-g'] = q_emb - gpt_emb
    feature_vectors['s-d'] = q_emb - ds_emb
    feature_vectors['g-d'] = gpt_emb - ds_emb

    feature_vectors['|s-g|'] = np.abs(q_emb - gpt_emb)
    feature_vectors['|s-d|'] = np.abs(q_emb - ds_emb)

    feature_vectors['|s-g|-|s-d|'] = np.abs(q_emb - gpt_emb) - np.abs(q_emb - ds_emb)

    if q_emb.shape[0] > 0 and q_emb.shape[1] < 1000:
        feature_vectors['s+g'] = np.hstack((q_emb, gpt_emb))
        feature_vectors['s+d'] = np.hstack((q_emb, ds_emb))
        feature_vectors['g+d'] = np.hstack((gpt_emb, ds_emb))
        feature_vectors['s+g+d'] = np.hstack((q_emb, gpt_emb, ds_emb))

    return feature_vectors

def hangisiIyi(data, model_name, model, tokenizer):
    analysis_data = data.copy()

    preference_column_name = "Hangisi iyi? (1: gpt4o daha iyi, 2: deepseek daha iyi, 3: ikisi de yeterince iyi, 4: ikisi de kötü)"


    analysis_data['Hangisi iyi?_numeric'] = pd.to_numeric(analysis_data[preference_column_name], errors='coerce')
    analysis_data = analysis_data.dropna(subset=[preference_column_name, 'Hangisi iyi?_numeric'])

    if analysis_data.empty:
        print(f"Filtreleme sonrası gecerli veri kalmadi. Siniflandirma atlanıyor.")
        return None

    required_cols = ['Soru', 'gpt4o cevabı', 'deepseek cevabı']

    question_embeddings = calculate_embeddings(analysis_data['Soru'].tolist(), model, tokenizer)
    gpt4o_answer_embeddings = calculate_embeddings(analysis_data['gpt4o cevabı'].tolist(), model, tokenizer)
    deepseek_answer_embeddings = calculate_embeddings(analysis_data['deepseek cevabı'].tolist(), model, tokenizer)

    if len(question_embeddings) == 0 or len(gpt4o_answer_embeddings) == 0 or len(deepseek_answer_embeddings) == 0:
        print(f"Hata: Embeddingler olusturulamadi. Analiz atlanıyor.")
        return None

    feature_vectors = create_feature_vectors(question_embeddings, gpt4o_answer_embeddings, deepseek_answer_embeddings)

    y = analysis_data['Hangisi iyi?_numeric'].values

    X_train, X_test, y_train, y_test = {}, {}, None, None
    first_split = True

    for feature_name, X in feature_vectors.items():
        if X.shape[0] == 0:
            print(f"Uyari: '{feature_name}' ozelligi icin ornek yok. Atlanıyor.")
            continue

        if first_split:
            unique_classes, class_counts = np.unique(y, return_counts=True)
            if X.shape[0] < 2 or len(y) < 2 or len(unique_classes) < 2 or np.any(class_counts < 2):
                 print(f"Uyari: Yetersiz ornek veya sinif sayisi. Tabakalamasiz ayriliyor.")
                 X_train[feature_name], X_test[feature_name], y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            else:
                X_train[feature_name], X_test[feature_name], y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            first_split = False
        else:
            unique_classes, class_counts = np.unique(y, return_counts=True)
            if X.shape[0] < 2 or len(y) < 2 or len(unique_classes) < 2 or np.any(class_counts < 2):
                 print(f"Uyari: Yetersiz ornek veya sinif sayisi. Tabakalamasiz ayriliyor.")
                 X_train[feature_name], X_test[feature_name], _, _ = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
            else:
                X_train[feature_name], X_test[feature_name], _, _ = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )

    results = {}

    if not X_train or y_train is None or len(y_train) == 0:
        print("Train/test ayrımı icin yeterli ornek veya etiket bulunamadi.")
        return {'model': model_name, 'siniflandirma_sonuclari': {}}

    for feature_name, X_train_feat in X_train.items():
        print(f"'{feature_name}' ozelligi ile siniflandirici egitiliyor...")

        try:
            classifier = LogisticRegression(max_iter=2000, random_state=42, solver='liblinear')
            classifier.fit(X_train_feat, y_train)

            X_test_feat = X_test[feature_name]
            y_pred = classifier.predict(X_test_feat)

            accuracy = accuracy_score(y_test, y_pred)

            report = None
            try:
                if len(y_test) > 0 and len(y_pred) > 0:
                    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                else:
                    print(f"Uyari: Bos test seti veya tahminler. Rapor olusturulamiyor.")
            except Exception as cr_e:
                print(f"Uyari: Siniflandirma raporu olusturulamadi: {cr_e}")

            results[feature_name] = {
                'accuracy': accuracy,
                'report': report
            }

            print(f"Ozellik: {feature_name}, Dogruluk: {accuracy:.4f}")

        except Exception as e:
            print(f"Hata: '{feature_name}' ozelligi icin siniflandirici egitilirken hata: {e}")
            results[feature_name] = {
                'accuracy': 0.0,
                'report': None,
                'error': str(e)
            }

    if results:
        plot_classification_results(results, model_name)

    return {
        'model': model_name,
        'siniflandirma_sonuclari': results
    }

def plot_classification_results(results, model_name):
    valid_results = {f: r for f, r in results.items() if r.get('report') is not None}

    features = list(valid_results.keys())
    accuracies = [valid_results[f]['accuracy'] for f in features]

    sorted_indices = np.argsort(accuracies)[::-1]
    sorted_features = [features[i] for i in sorted_indices]
    sorted_accuracies = [accuracies[i] for i in sorted_indices]

    df_plot = pd.DataFrame({
        'Ozellik': sorted_features, 
        'Dogruluk': sorted_accuracies, 
        'Model': [model_name] * len(sorted_features)  
    })

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Ozellik', y='Dogruluk', data=df_plot)  
    plt.title(f'Ozellik Tipine Gore Siniflandirma Dogrulugu ({model_name})')
    plt.xlabel('Ozellik Vektoru Tipi')
    plt.ylabel('Dogruluk')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.tight_layout()

    ax = plt.gca()
    bars = ax.patches

    for bar, acc in zip(bars, sorted_accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{acc:.3f}', ha='center', va='bottom')

    plt.savefig(f'{model_name}_siniflandirma_dogrulugu.png')
    plt.close()

def modelKarsilastir1(results_list):
    results_list = [r for r in results_list if r is not None]

    modeller = [r['model'] for r in results_list]
    gpt4o_top1 = [r['gpt4o_top1'] for r in results_list]
    gpt4o_top5 = [r['gpt4o_top5'] for r in results_list]
    deepseek_top1 = [r['deepseek_top1'] for r in results_list]
    deepseek_top5 = [r['deepseek_top5'] for r in results_list]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    x = np.arange(len(modeller))
    genislik = 0.35

    bars1_1 = ax1.bar(x - genislik/2, gpt4o_top1, genislik, label='GPT4o')
    bars1_2 = ax1.bar(x + genislik/2, deepseek_top1, genislik, label='Deepseek')
    ax1.set_title('Modele Gore Top-1 Basarisi')
    ax1.set_xticks(x)
    ax1.set_xticklabels(modeller)
    ax1.set_ylabel('Basari')
    ax1.legend()
    ax1.set_ylim(0, 1.1)

    for cubuk in bars1_1:
        yDegeri = cubuk.get_height()
        ax1.text(cubuk.get_x() + cubuk.get_width()/2, yDegeri + 0.01, f'{yDegeri:.3f}', ha='center', va='bottom')
    for cubuk in bars1_2:
        yDegeri = cubuk.get_height()
        ax1.text(cubuk.get_x() + cubuk.get_width()/2, yDegeri + 0.01, f'{yDegeri:.3f}', ha='center', va='bottom')

    bars2_1 = ax2.bar(x - genislik/2, gpt4o_top5, genislik, label='GPT4o')
    bars2_2 = ax2.bar(x + genislik/2, deepseek_top5, genislik, label='Deepseek')
    ax2.set_title('Modele Gore Top-5 Basarisi')
    ax2.set_xticks(x)
    ax2.set_xticklabels(modeller)
    ax2.set_ylabel('Basari')
    ax2.legend()
    ax2.set_ylim(0, 1.1)

    for cubuk in bars2_1:
        yDegeri = cubuk.get_height()
        ax2.text(cubuk.get_x() + cubuk.get_width()/2, yDegeri + 0.01, f'{yDegeri:.3f}', ha='center', va='bottom')
    for cubuk in bars2_2:
        yDegeri = cubuk.get_height()
        ax2.text(cubuk.get_x() + cubuk.get_width()/2, yDegeri + 0.01, f'{yDegeri:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('embedding_modelleri_karsilastirmasi_sorudan_cevaba.png')
    plt.close()

def modelKarsilastir2(results_list):
    results_list = [r for r in results_list if r is not None and r.get('siniflandirma_sonuclari')]

    karsilastirma_verisi = []

    ozellik_basari_sayisi = {}
    tum_ozellikler = set()

    for sonuc in results_list:
        for ozellik, res in sonuc['siniflandirma_sonuclari'].items():
            tum_ozellikler.add(ozellik)
            if res.get('report') is not None:
                ozellik_basari_sayisi[ozellik] = ozellik_basari_sayisi.get(ozellik, 0) + 1

    ortak_ozellikler = {f for f, sayi in ozellik_basari_sayisi.items() if sayi >= 2}

    temel_ozellikler = {'s', 'g', 'd', 's-g', 's-d', 'g-d', '|s-g|', '|s-d|', '|s-g|-|s-d|'}
    for bo in temel_ozellikler:
        if bo in tum_ozellikler and ozellik_basari_sayisi.get(bo, 0) >= 1:
             ortak_ozellikler.add(bo)

    onemli_ozellikler = sorted(list(ortak_ozellikler))
    if len(onemli_ozellikler) > 15:
         onemli_ozellikler = onemli_ozellikler[:15]

    for sonuc in results_list:
        model = sonuc['model']
        siniflandirma_sonuclari = sonuc['siniflandirma_sonuclari']

        for ozellik in onemli_ozellikler:
            if ozellik in siniflandirma_sonuclari and siniflandirma_sonuclari[ozellik].get('report') is not None:
                dogruluk = siniflandirma_sonuclari[ozellik]['accuracy']
                karsilastirma_verisi.append({
                    'Model': model,
                    'Ozellik': ozellik,
                    'Dogruluk': dogruluk
                })

    df_karsilastirma = pd.DataFrame(karsilastirma_verisi)
    print(f"Ortak ozellikler cizdiriliyor {onemli_ozellikler}")

    plt.figure(figsize=(15, 8))
    sns.barplot(x='Ozellik', y='Dogruluk', hue='Model', data=df_karsilastirma)
    plt.title('Ozellik Tipine ve Embedding Modeline Gore Siniflandirma Dogrulugu')
    plt.xlabel('Ozellik Vektoru Tipi')
    plt.ylabel('Dogruluk')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.legend(title='Embedding Modeli')
    plt.tight_layout()
    plt.savefig('embedding_modelleri_karsilastirmasi_hangisi_iyi.png')
    plt.close()

    if not df_karsilastirma.empty:
        pivot_tablo = df_karsilastirma.pivot(index='Ozellik', columns='Model', values='Dogruluk')
        print("\nSiniflandirma dogruluk karsilastirma tablosu")
        print(pivot_tablo)

        pivot_tablo.to_csv('embedding_modelleri_karsilastirma_tablosu.csv')

def main():
    tumVeri = pd.read_excel('ogrenci_sorular_2025.xlsx')
    print(f"toplam satir sayisi {len(tumVeri)}")
    veri1 = tumVeri.sample(n=1000, random_state=42).copy()
    print(f"Ilk soru icin 1000 ornek secildi.")
    veri2 = tumVeri.copy()
    print("Ikinci soru icin tum veri kullanilacak.")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    modeller = [("Jina", "jinaai/jina-embeddings-v3"), ("CosmosE5", "ytu-ce-cosmos/turkish-e5-large"),("E5", "intfloat/multilingual-e5-large-instruct")]
    mevcutModeller = []
    
    for modelAdi, modelPath in modeller:
        print(f"{modelAdi} modeli yukleniyor.")
        if "jina" in modelAdi.lower():
            model, tokenizer = modelYukle(modelPath)
            mevcutModeller.append((modelAdi, model, tokenizer))
            print(f"{modelAdi} yuklendi")
        else:
            model, tokenizer = modelYukle(modelPath)
            mevcutModeller.append((modelAdi, model, tokenizer))
            print(f"'{modelAdi}' basariyla yuklendi.")

    if not mevcutModeller:
        print("Hicbir model yuklenemedi.")
        return

    soru1sonculari = []
    soru2sonuclari = []

    for model_adi, model, tokenizer in mevcutModeller:
        sonuclar_a = sorudanCevaba(veri1, modelAdi, model, tokenizer)
        if sonuclar_a is not None:
            soru1sonculari.append(sonuclar_a)

        sonuclar_b = hangisiIyi(veri2, modelAdi, model, tokenizer)
        if sonuclar_b is not None:
            soru2sonuclari.append(sonuclar_b)


    if len(soru1sonculari) > 1:
        print("\nSorudan cevaba sonuclari karsilastiriliyor...")
        modelKarsilastir1(soru1sonculari)

    if len(soru2sonuclari) > 1:
        print("\nHangisi iyi sonuclari karsilastiriliyor.")
        modelKarsilastir2(soru2sonuclari)


if __name__ == "__main__":
    main()