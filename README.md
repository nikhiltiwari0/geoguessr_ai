# GEO GUESSR AI.

This was made during MadData Hackathon 2025

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Frontend Setup](#frontend-setup)
- [Backend Setup](#backend-setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

GEO GUESSR AI is a fun and interactive game that allows users to input an image, and our ML model will try to guess what country the user sent a photo of. This project was developed during the MadData Hackathon 2025.

## Installation

To set up the project locally, follow these steps:

### Frontend Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/nikhiltiwari0/geo-guessr-ai.git
   cd geo-guessr-ai
   ```

2. **Navigate to the frontend directory:**

   ```bash
   cd frontend
   ```

3. **Install dependencies:**
   Make sure you have [Node.js](https://nodejs.org/) installed, then run:

   ```bash
   npm install
   npm install --save leaflet
   ```

4. **Start the frontend application:**

   ```bash
   npm run dev
   ```

### Backend Setup

1. **Navigate to the backend directory:**

   ```bash
   cd backend
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   Make sure you have [Python](https://www.python.org/downloads/) installed, then run:

   ```bash
   pip install -r requirements.txt
   ```

4. **Start the backend application:**

   ```bash
   python app.py
   ```

## Usage

Once both the frontend and backend applications are running, open your browser and navigate to `http://localhost:5173` to start playing!
