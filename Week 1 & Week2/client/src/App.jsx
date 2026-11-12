import { useEffect, useState } from 'react';
import axios from 'axios';

function App() {
  const [todos, setTodos] = useState([]);
  const [text, setText] = useState('');
  const [error, setError] = useState('');

  const completedCount = todos.filter((todo) => todo.completed).length;

  const fetchTodos = async () => {
    const { data } = await axios.get('/api/todos');
    setTodos(data);
  };

  useEffect(() => {
    fetchTodos();
  }, []);

  const addTodo = async (e) => {
    e.preventDefault();
    const trimmedText = text.trim();
    if (!trimmedText) return;

    const taskExists = todos.some(
      (todo) => todo.text.trim().toLowerCase() === trimmedText.toLowerCase()
    );
    if (taskExists) {
      setError('Task already exists');
      return;
    }

    try {
      await axios.post('/api/todos', { text: trimmedText });
      setText('');
      setError('');
      fetchTodos();
    } catch (err) {
      if (err.response?.status === 409) {
        setError('Task already exists');
      } else {
        setError('Failed to add task');
      }
    }
  };

  const toggleTodo = async (todo) => {
    await axios.put(`/api/todos/${todo._id}`, { completed: !todo.completed });
    fetchTodos();
  };

  const toggleComplete = async (todo, checked) => {
    await axios.put(`/api/todos/${todo._id}`, { completed: checked });
    fetchTodos();
  };

  const [editingId, setEditingId] = useState(null);
  const [editingText, setEditingText] = useState('');

  const formatDate = (value) => {
    if (!value) return '';
    const date = new Date(value);
    return date.toLocaleString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    });
  };

  const startEdit = (todo) => {
    setEditingId(todo._id);
    setEditingText(todo.text);
  };

  const saveEdit = async (e) => {
    e.preventDefault();
    if (!editingText.trim()) return;
    await axios.put(`/api/todos/${editingId}`, { text: editingText });
    setEditingId(null);
    setEditingText('');
    fetchTodos();
  };

  const cancelEdit = () => {
    setEditingId(null);
    setEditingText('');
  };

  const deleteTodo = async (id) => {
    await axios.delete(`/api/todos/${id}`);
    fetchTodos();
  };

  return (
    <div className="app">
      <div className="top-panel">
        <div>
          <p className="eyebrow">Task manager</p>
          <h1>MERN To-Do List</h1>
          <p className="subtitle">Organize your day with style. Add tasks, track progress, and stay productive.</p>
        </div>
        <div className="summary-card">
          <span>{todos.length} tasks</span>
          <strong>{completedCount} completed</strong>
        </div>
      </div>

      <form onSubmit={addTodo} className="task-form">
        <input
          value={text}
          onChange={(e) => {
            setText(e.target.value);
            if (error) setError('');
          }}
          placeholder="Add a new task"
        />
        <button type="submit" className="primary-btn" disabled={!text.trim()}>
          Add task
        </button>
      </form>
      {error && <p className="error">{error}</p>}

      {todos.length > 0 ? (
        <>
          <div className="progress-row">
            <div className="progress-label">Progress</div>
            <div className="progress-track">
              <div
                className="progress-fill"
                style={{ width: `${Math.round((completedCount / todos.length) * 100)}%` }}
              />
            </div>
            <span className="progress-text">{completedCount}/{todos.length} done</span>
          </div>

          <ul className="todo-list">
            {todos.map((todo) => (
              <li key={todo._id} className={`todo-row ${todo.completed ? 'completed-row' : ''}`}>
                <div className="left">
                  <label className="checkbox-wrapper">
                    <input
                      type="checkbox"
                      checked={!!todo.completed}
                      onChange={(e) => toggleComplete(todo, e.target.checked)}
                    />
                    <span className="checkmark" />
                  </label>

                  {editingId === todo._id ? (
                    <form onSubmit={saveEdit} className="edit-form">
                      <input
                        className="edit-input"
                        value={editingText}
                        onChange={(e) => setEditingText(e.target.value)}
                        autoFocus
                      />
                      <button type="submit" className="small-btn">Save</button>
                      <button type="button" className="small-btn secondary" onClick={cancelEdit}>Cancel</button>
                    </form>
                  ) : (
                    <div className="todo-content">
                      <span className={`todo-text ${todo.completed ? 'completed' : ''}`}>{todo.text}</span>
                      <span className="todo-meta">Created {formatDate(todo.createdAt)}</span>
                    </div>
                  )}
                </div>

                <div className="actions">
                  {editingId !== todo._id && (
                    <>
                      <button onClick={() => startEdit(todo)} className="small-btn secondary">Edit</button>
                      <button onClick={() => deleteTodo(todo._id)} className="small-btn danger">Delete</button>
                    </>
                  )}
                </div>
              </li>
            ))}
          </ul>
        </>
      ) : (
        <div className="empty-state">
          <p>No tasks yet. Add your first task to get started.</p>
          <div className="empty-visual" />
        </div>
      )}
    </div>
  );
}

export default App;
