# Vagner Morais
## MVP - Qualidade de Software, Segurança e Sistemas Inteligentes


## Ferramenta de Análise de Score de Clientes
---

Este projeto foi desenvolvido com as seguintes tecnologias:


### Backend - Flask

Será necessário ter todas as libs python listadas no `requirements.txt` instaladas.
Após clonar o repositório, é necessário ir ao diretório raiz, pelo terminal, para poder executar os comandos descritos abaixo.

> É fortemente indicado o uso de ambientes virtuais do tipo [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html).

Para armazenar arquivos grandes como o do modelo, utilizamos o LFS (Git Large File Storage ).  
Download: https://git-lfs.com/
Instale o LFS: git lfs install  

```
(env)$ pip install -r requirements.txt
```

Este comando instala as dependências/bibliotecas, descritas no arquivo `requirements.txt`.

Para executar a API  basta executar:

```
(env)$ flask run --host 0.0.0.0 --port 5000
```

Em modo de desenvolvimento é recomendado executar utilizando o parâmetro reload, que reiniciará o servidor
automaticamente após uma mudança no código fonte. 

```
(env)$ flask run --host 0.0.0.0 --port 5000 --reload
```

Abra a API em  [http://localhost:5000/#/](http://localhost:5000/#/) no navegador para verificar o status da API em execução.

-----------------------------

## Para executar os testes, execute o comando "pytest"