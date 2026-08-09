# MERN To-Do List

A clean, full-stack task manager built with the MERN stack. Create tasks, mark them complete, edit or delete them, and track your progress from a responsive React interface.

## Features

- Create, edit, complete, and delete tasks
- Prevent duplicate task names (case-insensitive)
- View task creation times and completion progress
- Store tasks in MongoDB when it is available
- Continue working with temporary in-memory storage if MongoDB is unavailable
- REST API powered by Express

## Tech stack

| Area | Technology |
| --- | --- |
| Frontend | React, Vite, Axios |
| Backend | Node.js, Express |
| Database | MongoDB, Mongoose |
| Development | Nodemon, Concurrently |

## Project structure

```text
.
|-- client/                 # React + Vite frontend
|   `-- src/
|       |-- App.jsx         # Task manager interface
|       `-- styles.css      # Application styling
|-- server/
|   `-- server.js           # Express API and data model
|-- package.json            # Workspace scripts
`-- README.md
```

## Prerequisites

- [Node.js](https://nodejs.org/) 18 or newer
- npm
- MongoDB (optional - the application uses in-memory storage when no database is available)

## Getting started

1. Clone the repository and enter the project directory.

   ```bash
   git clone <your-repository-url>
   cd <repository-folder>
   ```

2. Install all workspace dependencies.

   ```bash
   npm install
   ```

3. Optionally create `server/.env` to configure the API and MongoDB connection.

   ```env
   PORT=5000
   MONGO_URI=mongodb://127.0.0.1:27017/todo-mern
   ```

4. Start the frontend and backend development servers together.

   ```bash
   npm run dev
   ```

5. Open [http://localhost:5173](http://localhost:5173) in your browser.

The frontend proxies `/api` requests to the backend at `http://localhost:5000`.

## Available scripts

| Command | Description |
| --- | --- |
| `npm run dev` | Run the client and server in development mode |
| `npm run build` | Create a production build of the React client |
| `npm run start` | Start the backend server |

## API reference

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/api/todos` | Get all tasks |
| `POST` | `/api/todos` | Create a task |
| `PUT` | `/api/todos/:id` | Update a task |
| `DELETE` | `/api/todos/:id` | Delete a task |

### Create a task

```http
POST /api/todos
Content-Type: application/json

{
  "text": "Finish project README",
  "dueDate": "2026-08-15T10:00:00.000Z"
}
```

`text` is required. `dueDate` is optional. Creating a duplicate task returns `409 Conflict`.

## Data persistence

When `MONGO_URI` is reachable, tasks are saved to MongoDB. If the connection fails, the API automatically switches to in-memory storage so the application remains usable. In-memory tasks are cleared whenever the server restarts.

## License

This project is available for educational use.
