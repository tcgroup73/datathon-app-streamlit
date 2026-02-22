import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==================== CONFIGURAÇÃO ====================
st.set_page_config(
    page_title='Previsão de Risco - Passos Mágicos',
    layout='wide',
    initial_sidebar_state='expanded'
)

# ==================== CARREGAMENTO DO MODELO ====================
@st.cache_resource
def carregar_modelo():
    model = joblib.load('modelo_risco_passos_magicos.pkl')
    features = joblib.load('features_model.pkl')
    return model, features

model, features = carregar_modelo()

# ==================== TEMA E ESTILO ====================
st.markdown("""
    <style>
        .header-main { font-size: 3em; color: #2E86AB; font-weight: bold; }
        .risk-high { color: #E63946; font-weight: bold; }
        .risk-low { color: #06A77D; font-weight: bold; }
        .metric-box { background-color: #F1F3F5; padding: 20px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

# ==================== TÍTULO E DESCRIÇÃO ====================
st.markdown('<p class="header-main">🔮 Previsão de Risco de Defasagem Escolar</p>', unsafe_allow_html=True)
st.write('Programa Passos Mágicos - Datathon 2026')
st.markdown("---")

# ==================== SIDEBAR ====================
st.sidebar.title("📊 Configurações")
st.sidebar.write("Preencha os dados do aluno para calcular o risco de defasagem.")

# ==================== FUNÇÃO DE PREVISÃO ====================
def prever_risco(input_data):
    """Realiza a previsão com o modelo treinado"""
    X = np.array(input_data).reshape(1, -1)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]
    return pred, prob

# ==================== ENTRADA DE DADOS ====================
st.sidebar.subheader("📝 Dados do Aluno")

input_values = {}
for feature in features:
    if feature == 'PEDRA_RANKING':
        input_values[feature] = st.sidebar.slider(
            f"{feature} (1=Quartzo, 2=Ágata, 3=Ametista, 4=Topázio)",
            min_value=1.0, max_value=4.0, step=0.1, value=2.0
        )
    elif feature == 'FASE':
        input_values[feature] = st.sidebar.slider(
            f"{feature} (Nível escolar)",
            min_value=1.0, max_value=12.0, step=0.5, value=7.0
        )
    elif feature == 'IDADE':
        input_values[feature] = st.sidebar.slider(
            f"{feature} (Anos)",
            min_value=6.0, max_value=25.0, step=0.5, value=14.0
        )
    else:
        input_values[feature] = st.sidebar.slider(
            f"{feature} (Indicador 0-10)",
            min_value=0.0, max_value=10.0, step=0.1, value=5.0
        )

# ==================== BOTÃO DE PREVISÃO ====================
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    calcular = st.button('🎯 Calcular Risco', use_container_width=True)

# ==================== EXIBIÇÃO DOS RESULTADOS ====================
if calcular:
    input_array = [input_values[feat] for feat in features]
    pred, prob = prever_risco(input_array)
    
    # Resultado principal
    st.markdown("---")
    st.subheader("📊 Resultado da Previsão")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if pred == 1:
            st.markdown('<p class="risk-high">🚨 EM RISCO DE DEFASAGEM</p>', unsafe_allow_html=True)
            status_color = "#E63946"
        else:
            st.markdown('<p class="risk-low">✅ EM FASE IDEAL</p>', unsafe_allow_html=True)
            status_color = "#06A77D"
    
    with col2:
        st.metric(
            label="Probabilidade de Risco",
            value=f"{prob*100:.1f}%",
            delta=None
        )
    
    with col3:
        risco_nivel = "Alto" if prob > 0.7 else "Médio" if prob > 0.4 else "Baixo"
        st.metric(
            label="Nível de Alerta",
            value=risco_nivel,
            delta=None
        )
    
    # Interpretação
    st.markdown("---")
    st.subheader("💡 Interpretação")
    
    if prob > 0.7:
        st.error(f"⚠️ **ALERTA ALTO**: Probabilidade de {prob*100:.1f}% - Intervenção imediata recomendada!")
        st.write("Recomendações:")
        st.write("• Aumentar acompanhamento pedagógico")
        st.write("• Avaliar indicadores psicossociais (IPS)")
        st.write("• Considerar reforço escolar ou acompanhamento psicológico")
    
    elif prob > 0.4:
        st.warning(f"⚠️ **ALERTA MÉDIO**: Probabilidade de {prob*100:.1f}% - Monitorar de perto")
        st.write("Recomendações:")
        st.write("• Acompanhamento regular")
        st.write("• Reforço em disciplinas específicas")
        st.write("• Feedback periódico com o aluno")
    
    else:
        st.success(f"✅ **BAIXO RISCO**: Probabilidade de {prob*100:.1f}% - Aluno em boa trajetória")
        st.write("Recomendações:")
        st.write("• Manter acompanhamento regular")
        st.write("• Incentivar continuidade e progressão")
    
    # Fatores Preditivos
    st.markdown("---")
    st.subheader("🔍 Fatores Mais Importantes para o Risco")
    
    importancias = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.write("**Ranking de Importância:**")
        for idx, (feat, imp) in enumerate(importancias.items(), 1):
            st.write(f"{idx}. **{feat}**: {imp*100:.1f}%")
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 5))
        importancias.plot(kind='barh', ax=ax, color='#2E86AB')
        ax.set_xlabel('Importância (%)', fontsize=10)
        ax.set_title('Influência das Variáveis no Risco', fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Dados resumidos
    st.markdown("---")
    st.subheader("📋 Dados Inseridos")
    
    df_inputs = pd.DataFrame({
        'Indicador': features,
        'Valor': input_array
    })
    st.dataframe(df_inputs, use_container_width=True)

# ==================== RODAPÉ ====================
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; font-size: 12px;'>
        Desenvolvido para Passos Mágicos | Datathon 2026<br>
        Modelo: Gradient Boosting Classifier | Acurácia: Validada em dados de 2022-2024
    </div>
""", unsafe_allow_html=True)