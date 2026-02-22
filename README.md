# 🎬 Sistema de Recomendação de Filmes com K-Means

Este projeto é um sistema de recomendação de filmes que utiliza o algoritmo de Machine Learning **K-Means** (Clusterização) para agrupar usuários com perfis e gostos semelhantes. A partir desse agrupamento, o sistema é capaz de sugerir novos filmes com base no que os usuários da mesma "tribo" mais gostaram.

A interface gráfica foi totalmente construída de forma interativa utilizando a biblioteca **Streamlit**.

---

## 🌐 Acesso Online (Live Demo)

Você pode testar e utilizar o sistema diretamente pelo seu navegador, sem precisar instalar nada no seu computador. A aplicação está hospedada na nuvem:

👉 **Acesse a aplicação clicando aqui:** [Sistema de Recomendação - K-Means](https://recommendation-system-using-clustering-2usjb3bcrsjin7pdzrpoen.streamlit.app/)

---

## 💻 Como executar o projeto localmente

Caso você queira baixar o código-fonte, modificar ou rodar o sistema na sua própria máquina, siga o tutorial abaixo.

### Pré-requisitos
* **Python 3.8+** instalado na máquina.
* Gerenciador de pacotes `pip`.

### Passo a Passo

#### 1. Clone o repositório ou baixe os arquivos

```bash
git clone https://github.com/SEU-USUARIO/recommendation-system-using-clustering.git
```

```bash
cd recommendation-system-using-clustering
```

*(Se você baixou o ZIP, basta extrair e abrir o terminal dentro da pasta extraída)*

#### 2. Instale as dependências do projeto
O projeto acompanha um arquivo `requirements.txt` com todas as bibliotecas necessárias. Para instalar, rode:

```bash
pip install -r requirements.txt
```

#### 3. Execute a aplicação via Streamlit
Após concluir a instalação das bibliotecas, inicie o servidor local executando o arquivo principal:

```bash
streamlit run app.py
```

#### 4. Acesse no Navegador
O Streamlit abrirá uma nova guia no seu navegador automaticamente. Caso isso não ocorra, acesse: `http://localhost:8501`.

---

## 📁 Estrutura do Projeto

O código foi modularizado para facilitar a manutenção e o entendimento. Aqui está a divisão dos arquivos principais:

* 📄 **`app.py`**: O arquivo principal da aplicação. Nele está contida toda a construção visual da interface (Dashboard, Menus, Abas e Gráficos), integrando os outros módulos.
* ⚙️ **`data_processing.py`**: Módulo responsável pela leitura das bases de dados originais, limpeza, mesclagem (Merge) e pelo cálculo percentual de proporção de gêneros consumidos por cada usuário.
* 🧠 **`ml_models.py`**: Contém a lógica de Machine Learning utilizando o `scikit-learn`. É responsável por treinar o modelo K-Means e gerar os gráficos de validação (Método do Cotovelo e Score da Silhueta).
* 🎯 **`recommender.py`**: O motor de recomendação. Avalia a qual cluster o usuário pertence, filtra os filmes que ele ainda não viu e calcula a popularidade e a nota média dentro do seu grupo para gerar as melhores indicações.
* 🗂️ **`/DataBase`**: Diretório que armazena os dados brutos (`movies.dat` e `ratings.dat`).
* 📜 **`requirements.txt`**: Lista das bibliotecas e dependências (ex: pandas, scikit-learn, streamlit, matplotlib).

---

## 🛠️ Tecnologias Utilizadas
* **Linguagem:** Python
* **Interface Web:** Streamlit
* **Manipulação de Dados:** Pandas
* **Machine Learning:** Scikit-Learn
* **Visualização Gráfica:** Matplotlib

---
Desenvolvido como projeto de estudo prático sobre Algoritmos de Clusterização (K-Means) e Sistemas de Recomendação.
