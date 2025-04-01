use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use std::{
    error::Error,
    io,
    process::Command,
    time::{Duration, Instant},
};
use tui::{
    backend::{Backend, CrosstermBackend},
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Span, Spans, Text},
    widgets::{Block, Borders, List, ListItem, Paragraph, Tabs, Wrap},
    Frame, Terminal,
};
use unicode_width::UnicodeWidthStr;

enum InputMode {
    Normal,
    Editing,
}

enum ActiveTab {
    Models,
    Chat,
    Settings,
}

struct App {
    input: String,
    input_mode: InputMode,
    active_tab: ActiveTab,
    models: Vec<String>,
    selected_model_index: Option<usize>,
    chat_history: Vec<ChatMessage>,
    should_quit: bool,
}

struct ChatMessage {
    sender: String,
    content: String,
    timestamp: String,
}

impl Default for App {
    fn default() -> App {
        App {
            input: String::new(),
            input_mode: InputMode::Normal,
            active_tab: ActiveTab::Models,
            models: Vec::new(),
            selected_model_index: None,
            chat_history: Vec::new(),
            should_quit: false,
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app state
    let mut app = App::default();
    
    // Load models on startup
    app.models = get_ollama_models().unwrap_or_else(|_| vec!["Error loading models".to_string()]);

    // Main loop
    let tick_rate = Duration::from_millis(250);
    let mut last_tick = Instant::now();
    
    loop {
        terminal.draw(|f| ui(f, &app))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        if event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                match app.input_mode {
                    InputMode::Normal => match key.code {
                        KeyCode::Char('q') => {
                            app.should_quit = true;
                        }
                        KeyCode::Char('e') => {
                            app.input_mode = InputMode::Editing;
                        }
                        KeyCode::Char('t') => {
                            // Toggle between tabs
                            app.active_tab = match app.active_tab {
                                ActiveTab::Models => ActiveTab::Chat,
                                ActiveTab::Chat => ActiveTab::Settings,
                                ActiveTab::Settings => ActiveTab::Models,
                            };
                        }
                        KeyCode::Down => {
                            if let ActiveTab::Models = app.active_tab {
                                if !app.models.is_empty() {
                                    app.selected_model_index = match app.selected_model_index {
                                        Some(i) => Some(if i >= app.models.len() - 1 { 0 } else { i + 1 }),
                                        None => Some(0),
                                    };
                                }
                            }
                        }
                        KeyCode::Up => {
                            if let ActiveTab::Models = app.active_tab {
                                if !app.models.is_empty() {
                                    app.selected_model_index = match app.selected_model_index {
                                        Some(i) => Some(if i == 0 { app.models.len() - 1 } else { i - 1 }),
                                        None => Some(app.models.len() - 1),
                                    };
                                }
                            }
                        }
                        KeyCode::Enter => {
                            if let ActiveTab::Models = app.active_tab {
                                if let Some(idx) = app.selected_model_index {
                                    app.active_tab = ActiveTab::Chat;
                                    let model_name = &app.models[idx];
                                    let message = format!("Selected model: {}", model_name);
                                    app.chat_history.push(ChatMessage {
                                        sender: "System".to_string(),
                                        content: message,
                                        timestamp: get_current_time(),
                                    });
                                }
                            }
                        }
                        _ => {}
                    },
                    InputMode::Editing => match key.code {
                        KeyCode::Enter => {
                            let query = app.input.drain(..).collect::<String>();
                            if !query.is_empty() {
                                app.chat_history.push(ChatMessage {
                                    sender: "You".to_string(),
                                    content: query.clone(),
                                    timestamp: get_current_time(),
                                });
                                
                                // Get response from ollama
                                if let Some(idx) = app.selected_model_index {
                                    let model_name = &app.models[idx];
                                    match get_ollama_response(model_name, &query) {
                                        Ok(response) => {
                                            app.chat_history.push(ChatMessage {
                                                sender: model_name.clone(),
                                                content: response,
                                                timestamp: get_current_time(),
                                            });
                                        }
                                        Err(e) => {
                                            app.chat_history.push(ChatMessage {
                                                sender: "Error".to_string(),
                                                content: format!("Failed to get response: {}", e),
                                                timestamp: get_current_time(),
                                            });
                                        }
                                    }
                                } else {
                                    app.chat_history.push(ChatMessage {
                                        sender: "System".to_string(),
                                        content: "Please select a model first from the Models tab.".to_string(),
                                        timestamp: get_current_time(),
                                    });
                                }
                            }
                            app.input_mode = InputMode::Normal;
                        }
                        KeyCode::Char(c) => {
                            app.input.push(c);
                        }
                        KeyCode::Backspace => {
                            app.input.pop();
                        }
                        KeyCode::Esc => {
                            app.input_mode = InputMode::Normal;
                        }
                        _ => {}
                    },
                }
            }
        }
        
        if app.should_quit {
            break;
        }
        
        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    Ok(())
}

fn ui<B: Backend>(f: &mut Frame<B>, app: &App) {
    // Create base layout
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints(
            [
                Constraint::Length(3),
                Constraint::Min(0),
                Constraint::Length(3),
            ]
            .as_ref(),
        )
        .split(f.size());

    // Create tabs
    let titles = vec!["Models", "Chat", "Settings"];
    let tabs = Tabs::new(titles.iter().map(|t| Spans::from(Span::styled(*t, Style::default().fg(Color::Green)))).collect())
        .block(Block::default().borders(Borders::ALL).title("Ollama TUI"))
        .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))
        .select(match app.active_tab {
            ActiveTab::Models => 0,
            ActiveTab::Chat => 1,
            ActiveTab::Settings => 2,
        });
    f.render_widget(tabs, chunks[0]);

    match app.active_tab {
        ActiveTab::Models => draw_models_tab(f, app, chunks[1]),
        ActiveTab::Chat => draw_chat_tab(f, app, chunks[1]),
        ActiveTab::Settings => draw_settings_tab(f, app, chunks[1]),
    }

    // Create input field
    let input = Paragraph::new(app.input.as_ref())
        .style(match app.input_mode {
            InputMode::Normal => Style::default(),
            InputMode::Editing => Style::default().fg(Color::Yellow),
        })
        .block(Block::default().borders(Borders::ALL).title("Input"));
    f.render_widget(input, chunks[2]);

    // Show cursor when in editing mode
    if let InputMode::Editing = app.input_mode {
        f.set_cursor(
            chunks[2].x + app.input.width() as u16 + 1,
            chunks[2].y + 1,
        );
    }
}

fn draw_models_tab<B: Backend>(f: &mut Frame<B>, app: &App, area: Rect) {
    let model_items: Vec<ListItem> = app
        .models
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let content = vec![Spans::from(Span::raw(m))];
            ListItem::new(content)
                .style(Style::default().fg(if Some(i) == app.selected_model_index {
                    Color::Yellow
                } else {
                    Color::White
                }))
        })
        .collect();

    let models_list = List::new(model_items)
        .block(Block::default().borders(Borders::ALL).title("Available Models"))
        .highlight_style(Style::default().add_modifier(Modifier::BOLD))
        .highlight_symbol(">> ");

    f.render_widget(models_list, area);
}

fn draw_chat_tab<B: Backend>(f: &mut Frame<B>, app: &App, area: Rect) {
    let chat_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0)].as_ref())
        .split(area);

    let mut formatted_chat = Vec::new();
    for message in &app.chat_history {
        formatted_chat.push(Spans::from(vec![
            Span::styled(
                format!("{} ({}): ", message.sender, message.timestamp),
                Style::default().fg(match message.sender.as_str() {
                    "You" => Color::Green,
                    "System" => Color::Blue,
                    "Error" => Color::Red,
                    _ => Color::Yellow,
                }).add_modifier(Modifier::BOLD),
            ),
        ]));
        
        // Split message content by lines for better formatting
        for line in message.content.lines() {
            formatted_chat.push(Spans::from(line.to_string()));
        }
        
        // Add spacing between messages
        formatted_chat.push(Spans::from(""));
    }

    let chat_paragraph = Paragraph::new(Text::from(formatted_chat))
        .block(Block::default().borders(Borders::ALL).title("Chat"))
        .wrap(Wrap { trim: false });

    f.render_widget(chat_paragraph, chat_layout[0]);
}

fn draw_settings_tab<B: Backend>(f: &mut Frame<B>, _app: &App, area: Rect) {
    let help_text = vec![
        Spans::from("Ollama TUI Help:"),
        Spans::from(""),
        Spans::from("- 'q': Quit the application"),
        Spans::from("- 't': Toggle between tabs"),
        Spans::from("- 'e': Enter edit mode for typing"),
        Spans::from("- Up/Down: Navigate model list"),
        Spans::from("- Enter: Select model or send message"),
        Spans::from("- Esc: Exit edit mode"),
        Spans::from(""),
        Spans::from("Server Info:"),
        Spans::from("- Port: 11434"),
    ];

    let settings = Paragraph::new(help_text)
        .block(Block::default().borders(Borders::ALL).title("Settings & Help"))
        .wrap(Wrap { trim: false });

    f.render_widget(settings, area);
}

fn get_ollama_models() -> Result<Vec<String>, Box<dyn Error>> {
    let output = Command::new("ollama")
        .arg("list")
        .output()?;
    
    if output.status.success() {
        let stdout = String::from_utf8(output.stdout)?;
        let models = stdout
            .lines()
            .skip(1) // Skip header line
            .filter_map(|line| {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if !parts.is_empty() {
                    Some(parts[0].to_string())
                } else {
                    None
                }
            })
            .collect();
        Ok(models)
    } else {
        Err("Failed to get ollama models".into())
    }
}

fn get_ollama_response(model: &str, prompt: &str) -> Result<String, Box<dyn Error>> {
    let output = Command::new("ollama")
        .arg("run")
        .arg(model)
        .arg(prompt)
        .output()?;
    
    if output.status.success() {
        let stdout = String::from_utf8(output.stdout)?;
        Ok(stdout)
    } else {
        let stderr = String::from_utf8(output.stderr)?;
        Err(format!("Failed to get response: {}", stderr).into())
    }
}

fn get_current_time() -> String {
    use chrono::Local;
    Local::now().format("%H:%M:%S").to_string()
}
