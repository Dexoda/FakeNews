# FakeNewsDetector

A system for detecting fake news based on linguistic analysis with a Telegram bot client interface.

## Overview

FakeNewsDetector is a comprehensive system that performs multi-level linguistic analysis on news texts to evaluate their credibility. The system uses statistical, linguistic, semantic, and structural analysis to identify signs of fake news, verifies information using external fact-checking services, and presents results through a Telegram bot.

## Features

- Multi-level text analysis (statistical, linguistic, semantic, structural)
- Fact verification through external APIs
- Information search across open sources
- Visualization of analysis results
- Telegram bot user interface
- Docker containerization

## System Requirements

- Docker and Docker Compose
- Internet access for external API calls
- Telegram Bot Token (required)
- Google Fact Check API Key (optional but recommended)

## Important Notes and Limitations

1. **API Keys**: For full functionality, you need to obtain the following API keys:
   - Telegram Bot Token (required): Get from [@BotFather](https://t.me/BotFather)
   - Google Fact Check API Key (optional): Get from [Google Cloud Console](https://console.cloud.google.com/)

2. **Simulated Features**: 
   - The open source search functionality is currently simulated for demonstration purposes and doesn't actually perform real searches.
   - ClaimReview API integration is also simulated.

3. **NLP Models**: On first startup, the system will download necessary NLP models which might take some time. Alternatively, you can run `setup_models.py` before starting the containers.

## Environment Variables

The system uses the following environment variables (defined in `.env` file):

| Variable | Description | Default Value |
|----------|-------------|---------------|
| USER_ID | UID for the tester user | 100001 |
| GROUP_ID | GID for the tester user | 100001 |
| TZ | Timezone | Europe/Moscow |
| DB_HOST | Database host | db |
| DB_PORT | Database port | 5432 |
| DB_USER | Database username | fakedetector |
| DB_PASSWORD | Database password | fakedetector |
| DB_NAME | Database name | fakedetector |
| API_HOST | API server host | 0.0.0.0 |
| API_PORT | API server port | 8000 |
| LOG_LEVEL | Logging level | INFO |
| TELEGRAM_BOT_TOKEN | Telegram bot token | (required) |
| GOOGLE_FACT_CHECK_API_KEY | Google Fact Check API key | (optional) |

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fakenewsdetector.git
   cd fakenewsdetector
   ```

2. Create and configure the environment variables file:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration, especially the TELEGRAM_BOT_TOKEN
   ```

3. Create and configure the configuration file:
   ```bash
   cp config.example.yml config.yml
   # Edit config.yml with your configuration
   ```

4. Get a Telegram Bot Token:
   - Contact [@BotFather](https://t.me/BotFather) on Telegram
   - Create a new bot with `/newbot` command
   - Copy the token to your `.env` file

5. (Optional) Get a Google Fact Check API Key:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project and enable the Fact Check Tools API
   - Create an API key and copy it to your `.env` file

6. Build and start the containers:
   ```bash
   docker-compose up -d
   ```

7. Check if the services are running:
   ```bash
   docker-compose ps
   ```

8. View the logs:
   ```bash
   docker-compose logs -f
   ```

## Usage

1. Start a chat with your Telegram bot
2. Send the bot a news article text or URL
3. Wait for the analysis to complete
4. Review the credibility assessment and detailed analysis results

## Project Structure

```
fakenewsdetector/
├── analyzer/              # Text analysis modules
│   ├── statistical.py     # Statistical analysis
│   ├── linguistic.py      # Linguistic analysis
│   ├── semantic.py        # Semantic analysis
│   ├── structural.py      # Structural analysis
│   └── pipeline.py        # Analysis pipeline
├── factcheck/             # Fact-checking modules
│   ├── api_client.py      # External API client
│   └── sources.py         # Open sources search
├── visualization/         # Visualization modules
│   ├── charts.py          # Chart generation
│   ├── heatmap.py         # Text heatmap generation
│   └── reports.py         # Report generation
├── app.py                 # Main server application
├── bot.py                 # Telegram bot
├── database.py            # Database interface
├── Dockerfile             # Docker configuration for server
├── Dockerfile.bot         # Docker configuration for bot
├── docker-compose.yml     # Docker Compose configuration
├── config.yml             # Application configuration
├── .env                   # Environment variables
├── .env.example           # Example environment variables file
├── setup_models.py        # Script to download NLP models
├── requirements.txt       # Python dependencies
└── README.md              # This documentation
```

## Technical Stack

- **Backend**: Python with FastAPI
- **Linguistic Analysis**: NLTK, spaCy, Dostoevsky (for Russian sentiment analysis)
- **Visualization**: Matplotlib, Seaborn
- **Telegram Bot**: python-telegram-bot
- **Database**: PostgreSQL
- **Containerization**: Docker, Docker Compose

## Development

To add new functionality or fix issues:

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and test them

3. Build and test with Docker:
   ```bash
   docker-compose build
   docker-compose up -d
   ```

4. Commit your changes:
   ```bash
   git commit -m "Add your feature description"
   ```

## Troubleshooting

### Common Issues

- **Bot not responding**: Check if the Telegram token is correct and the bot container is running
- **Analysis taking too long**: On first run, the system needs to download NLP models which may take time
- **Container not starting**: Check Docker logs and verify configuration files
- **Missing NLP models**: If you encounter errors about missing models, run `setup_models.py` manually

### Checking Logs

```bash
# Check logs for the API server
docker-compose logs api-server

# Check logs for the bot
docker-compose logs bot

# Check logs for the database
docker-compose logs db
```

## License

[MIT License](LICENSE)
