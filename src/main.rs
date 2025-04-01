use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{prelude::*, widgets::*, text::{Line, Span}};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{
    error::Error,
    io::{self},
    time::{Duration},
};
use thiserror::Error;
use tokio::sync::mpsc;
use futures::StreamExt;
use pulldown_cmark::{Event as MDEvent, Options, Parser, Tag as MDTag};

const OLLAMA_BASE_URL: &str = "http://127.0.0.1:11434";

// --- Error Handling ---
#[derive(Error, Debug)]
enum AppError {
    #[error("IO Error: {0}")]
    Io(#[from] io::Error),
    #[error("API Request Error: {0}")]
    ApiRequest(#[from] reqwest::Error),
    #[error("API Response Error: {0}")]
    ApiResponse(String),
    #[error("JSON Error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Channel Send Error: {0}")]
    ChannelSend(String),
    #[error("Channel Receive Error")]
    ChannelReceive,
}

// --- Ollama API Structures ---
#[derive(Serialize)]
struct OllamaGenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
}

#[derive(Deserialize, Debug)]
struct OllamaGenerateChunk {
    model: String,
    created_at: String,
    response: String,
    done: bool,
    // context: Option<Vec<i64>>,
    // total_duration: Option<u64>,
    // load_duration: Option<u64>,
    // prompt_eval_count: Option<usize>,
    // prompt_eval_duration: Option<u64>,
    // eval_count: Option<usize>,
    // eval_duration: Option<u64>,
}


#[derive(Deserialize, Debug)]
struct OllamaTagsResponse {
    models: Vec<ModelInfo>,
}

#[derive(Deserialize, Debug, Clone)]
struct ModelInfo {
    name: String,
    // modified_at: String,
    // size: u64,
    // digest: String,
    // details: ModelDetails,
}

// --- Application State ---
enum InputMode {
    Normal,
    Editing,
}

struct ConversationTurn {
    sender: String,
    text: String,
}

struct App {
    input_mode: InputMode,
    input_buffer: String,
    conversation: Vec<ConversationTurn>,
    models: Vec<String>,
    selected_model_index: Option<usize>,
    is_loading: bool,
    status_message: String,
    scroll_offset: u16,
    http_client: Client,
    event_receiver: mpsc::Receiver<AppEvent>,
    event_sender: mpsc::Sender<AppEvent>,
}

// --- Events for Async Communication ---
#[derive(Debug)]
enum AppEvent {
    ModelsFetched(Result<Vec<String>, AppError>),
    OllamaChunk(String),
    OllamaDone,
    OllamaError(String),
}

impl App {
    fn new(rx: mpsc::Receiver<AppEvent>, tx: mpsc::Sender<AppEvent>) -> Self {
        App {
            input_mode: InputMode::Normal,
            input_buffer: String::new(),
            conversation: Vec::new(),
            models: Vec::new(),
            selected_model_index: None,
            is_loading: false,
            status_message: "Fetching models...".to_string(),
            scroll_offset: 0,
            http_client: Client::new(),
            event_receiver: rx,
            event_sender: tx,
        }
    }

    fn get_selected_model_name(&self) -> Option<String> {
        self.selected_model_index
            .and_then(|index| self.models.get(index).cloned())
    }

    fn submit_prompt(&mut self) {
        if let Some(model_name) = self.get_selected_model_name() {
            if !self.input_buffer.is_empty() {
                let prompt = self.input_buffer.trim().to_string();
                if prompt.is_empty() {
                     self.status_message = "Cannot send an empty prompt.".to_string();
                     self.input_mode = InputMode::Normal;
                     return;
                }

                self.conversation.push(ConversationTurn {
                    sender: "You".to_string(),
                    text: prompt.clone(),
                });
                self.input_buffer.clear();
                self.is_loading = true;
                self.status_message = format!("Asking {}...", model_name);
                self.scroll_offset = 0;

                let client = self.http_client.clone();
                let event_sender = self.event_sender.clone();
                tokio::spawn(async move {
                    // We handle errors inside stream_ollama_response by sending AppEvents
                    // So we don't necessarily need to handle the task result here unless it panics
                    let _ = stream_ollama_response(client, model_name, prompt, event_sender.clone()).await;
                });
            } else {
                // Buffer is empty or only whitespace
                 self.status_message = "Cannot send an empty prompt.".to_string();
            }
        } else {
            self.status_message = "Error: No model selected.".to_string();
        }
        // Always return to Normal mode after trying to submit
        self.input_mode = InputMode::Normal;
    }


    fn scroll_down(&mut self, amount: u16) {
        self.scroll_offset = self.scroll_offset.saturating_add(amount);
        // TODO: Clamp scroll_offset based on actual content height for more robust scrolling
    }

    fn scroll_up(&mut self, amount: u16) {
        self.scroll_offset = self.scroll_offset.saturating_sub(amount);
    }
}

// --- Main Application Logic ---
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create communication channel and App instance
    let (tx, rx) = mpsc::channel(100); // Channel buffer size 100
    let mut app = App::new(rx, tx.clone());

    // --- Initial Async Tasks ---
    // Fetch models immediately
    let client = app.http_client.clone();
    let initial_event_sender = app.event_sender.clone();
    tokio::spawn(async move {
        let models_result = fetch_models(client).await;
        // Send result back, handling potential channel send error
        if initial_event_sender.send(AppEvent::ModelsFetched(models_result)).await.is_err() {
            // Error sending back to main loop, maybe log to stderr
            eprintln!("Error: Failed to send fetched models back to main loop.");
        }
    });

    // Run the main TUI loop
    let res = run_app(&mut terminal, &mut app).await;

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    // Print errors if the app loop returned an error
    if let Err(err) = res {
        eprintln!("TUI Error: {}", err);
    }

    Ok(())
}


// --- Main Event Loop ---
async fn run_app<B: Backend>(
    terminal: &mut Terminal<B>,
    app: &mut App,
) -> Result<(), AppError> {

    loop {
        // Draw UI on each iteration
        terminal.draw(|f| ui(f, app))?;

        // --- Handle Input Events (non-blocking) ---
        // Poll for crossterm events with a small timeout
        if event::poll(Duration::from_millis(50))? {
            // If poll is true, read() is guaranteed not to block
            match event::read()? {
                // Process only key press events
                Event::Key(key) if key.kind == KeyEventKind::Press => {
                    match app.input_mode {
                        InputMode::Normal => match key.code {
                            KeyCode::Char('q') => return Ok(()), // Quit
                            KeyCode::Char('j') | KeyCode::Down => {
                                if !app.models.is_empty() {
                                    let i = app.selected_model_index.unwrap_or(0);
                                    let next = if i >= app.models.len() - 1 { 0 } else { i + 1 };
                                    app.selected_model_index = Some(next);
                                }
                            },
                            KeyCode::Char('k') | KeyCode::Up => {
                                if !app.models.is_empty() {
                                    let i = app.selected_model_index.unwrap_or(0);
                                    let prev = if i == 0 { app.models.len() - 1 } else { i - 1 };
                                    app.selected_model_index = Some(prev);
                                }
                            },
                            KeyCode::Enter => {
                                if app.selected_model_index.is_some() {
                                    app.input_mode = InputMode::Editing;
                                    app.status_message = "Editing prompt... Enter: Newline, Ctrl+S: Send, Esc: Cancel.".to_string();
                                } else {
                                    app.status_message = "Select a model first (Up/Down keys).".to_string();
                                }
                            }
                            KeyCode::PageDown => app.scroll_down(10), // Scroll faster
                            KeyCode::PageUp => app.scroll_up(10),   // Scroll faster
                            _ => {} // Ignore other keys in Normal mode
                        },
                        InputMode::Editing => match (key.code, key.modifiers) {
                            // Use Ctrl+S to send the prompt
                            (KeyCode::Char('s'), KeyModifiers::CONTROL) => {
                                app.submit_prompt(); // Handles state change and status message
                            }
                            // Enter key inserts a newline
                             (KeyCode::Enter, _) => {
                                app.input_buffer.push('\n');
                            }
                            // Regular character input (handle Shift implicitly)
                            (KeyCode::Char(c), modifier) if modifier == KeyModifiers::NONE || modifier == KeyModifiers::SHIFT => {
                                app.input_buffer.push(c);
                            }
                            // Backspace removes the last character
                            (KeyCode::Backspace, _) => {
                                app.input_buffer.pop();
                            }
                            // Escape cancels editing and clears the buffer
                            (KeyCode::Esc, _) => {
                                app.input_mode = InputMode::Normal;
                                app.input_buffer.clear(); // Clear buffer on cancel
                                app.status_message = "Input cancelled. Press 'Enter' to start typing again.".to_string();
                            }
                            _ => {} // Ignore other keys/modifiers in Editing mode
                        }
                    }
                }
                // Handle terminal resize events if necessary (redraw is automatic)
                Event::Resize(_, _) => {}
                // Ignore other event types (Mouse, Focus, Paste, etc.)
                _ => {}
            }
        }

        // --- Handle Async Events from Ollama tasks (non-blocking) ---
        match app.event_receiver.try_recv() {
            Ok(app_event) => {
                 // Process received AppEvent
                 match app_event {
                    AppEvent::ModelsFetched(Ok(models)) => {
                        app.models = models;
                        if !app.models.is_empty() {
                            app.selected_model_index = Some(0); // Select first model
                            app.status_message = format!(
                                "{} models loaded. Select: Up/Down, Chat: Enter (then Ctrl+S to send)",
                                app.models.len()
                            );
                        } else {
                            app.status_message = "No models found on Ollama server.".to_string();
                        }
                    }
                    AppEvent::ModelsFetched(Err(e)) => {
                        // Display error fetching models
                        app.status_message = format!("Error fetching models: {}", e);
                        // Optionally add to conversation log
                        app.conversation.push(ConversationTurn {
                            sender: "System Error".to_string(),
                            text: format!("Failed to fetch models: {}", e)
                        });
                    }
                    AppEvent::OllamaChunk(chunk) => {
                        // Append chunk to the last conversation turn if it's from the model
                        let model_name = app.get_selected_model_name().unwrap_or_else(|| "Model".to_string());
                        if let Some(last_turn) = app.conversation.last_mut() {
                            if last_turn.sender == model_name {
                                last_turn.text.push_str(&chunk); // Append to existing model response
                            } else {
                                // Last turn was from User or Error, start new Model turn
                                app.conversation.push(ConversationTurn {
                                    sender: model_name,
                                    text: chunk,
                                });
                            }
                        } else {
                            // Conversation is empty, start the first Model turn
                            app.conversation.push(ConversationTurn {
                                sender: model_name,
                                text: chunk,
                            });
                        }
                        // TODO: Implement auto-scrolling logic if desired
                    }
                    AppEvent::OllamaDone => {
                        // Mark loading as finished, update status
                        app.is_loading = false;
                        app.status_message = "Response received. Press 'Enter' to type (Ctrl+S to send).".to_string();
                    }
                    AppEvent::OllamaError(err_msg) => {
                        // Mark loading finished, display error
                        app.is_loading = false;
                        // Add error to conversation for visibility
                        app.conversation.push(ConversationTurn {
                            sender: "Error".to_string(),
                            text: err_msg.clone()
                        });
                        // Update status bar
                        app.status_message = format!("Error occurred: {}", err_msg);
                    }
                }
            }
            // No message received from async tasks
            Err(mpsc::error::TryRecvError::Empty) => {}
            // Channel disconnected - critical error
            Err(mpsc::error::TryRecvError::Disconnected) => {
                app.status_message = "Critical Error: Async event channel disconnected.".to_string();
                terminal.draw(|f| ui(f, app))?; // Draw final error before exiting
                return Err(AppError::ChannelReceive);
            }
        }

        // Short sleep to prevent high CPU usage when idle
        tokio::time::sleep(Duration::from_millis(10)).await;

    } // End main loop
}


// --- UI Drawing Logic ---
// Takes immutable borrow of App as state changes happen in run_app loop
fn ui(f: &mut Frame, app: &App) {
    // Main layout: Models List | Right Pane
    let main_chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(25), Constraint::Percentage(75)].as_ref())
        .split(f.size());

    // --- Left side: Models List ---
     let model_items: Vec<ListItem> = app
        .models
        .iter()
        .map(|m| ListItem::new(m.as_str())) // Creates ListItem<'a> borrowing from app.models
        .collect();

    let models_list = List::new(model_items) // List holds Vec<ListItem<'a>>
        .block(Block::default().borders(Borders::ALL).title(" Models (j/k) "))
        .highlight_style(
            Style::default()
                .add_modifier(Modifier::BOLD)
                .bg(Color::Blue),
        )
        .highlight_symbol("> ");

    // ListState needs to be mutable for rendering selection
    let mut list_state = ListState::default();
    list_state.select(app.selected_model_index);

    f.render_stateful_widget(models_list, main_chunks[0], &mut list_state);

    // --- Right side: Conversation, Input, Status ---
    let right_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(
            [
                Constraint::Min(1),        // Conversation area
                Constraint::Length(7),     // Input area height
                Constraint::Length(1),     // Status bar height
            ]
            .as_ref(),
        )
        .split(main_chunks[1]);

    // --- Conversation Area ---
    // Build the content for the conversation paragraph
    let mut conversation_content: Vec<Line> = Vec::new();
    for turn in &app.conversation { // Borrow each turn
         let prefix_style = match turn.sender.as_str() {
            "You" => Style::default().fg(Color::Green).add_modifier(Modifier::BOLD),
            "Error" | "System Error" => Style::default().fg(Color::Red).add_modifier(Modifier::BOLD),
            _ => Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        };
         // Create Line<'a> borrowing from turn.sender
         let prefix = Line::styled(format!("{}: ", turn.sender), prefix_style);
         conversation_content.push(prefix);

         // Render the turn's text using Markdown
         // render_markdown returns Vec<Line<'a>> borrowing from turn.text
         conversation_content.extend(render_markdown(&turn.text));

         // Add spacing between turns
         conversation_content.push(Line::from("")); // Creates Line<'static>
    }

    let conversation_paragraph = Paragraph::new(conversation_content) // Takes Vec<Line<'_>>
        .block(Block::default().borders(Borders::ALL).title(" Conversation (PgUp/PgDn) "))
        .wrap(Wrap { trim: false }) // Don't trim whitespace on wrapped lines
        .scroll((app.scroll_offset, 0)); // Apply vertical scroll offset

    f.render_widget(conversation_paragraph, right_chunks[0]);

    // --- Input Area ---
    let input_title = match app.input_mode {
        InputMode::Editing => " Input (Enter: Newline, Ctrl+S: Send, Esc: Cancel) ",
        InputMode::Normal => " Input (Press Enter to type) ",
    };
    let input_block_style = match app.input_mode {
        InputMode::Editing => Style::default().fg(Color::Yellow), // Highlight border
        InputMode::Normal => Style::default(),
    };
    let input_block = Block::default()
        .borders(Borders::ALL)
        .title(input_title)
        .border_style(input_block_style);

    // Create the input paragraph, borrowing from app.input_buffer
    let input_paragraph = Paragraph::new(app.input_buffer.as_str()) // Creates Paragraph<'a>
        .block(input_block)
        .wrap(Wrap { trim: false }); // Wrap long input lines

    f.render_widget(input_paragraph, right_chunks[1]);

    // Set cursor position visually only when editing
    if let InputMode::Editing = app.input_mode {
        let input_area = right_chunks[1];
        let buffer_char_count = app.input_buffer.chars().count();
        // Calculate width inside borders, ensure it's at least 1
        let input_width = input_area.width.saturating_sub(2).max(1);

        // Estimate row/col (doesn't perfectly handle complex wrapping)
        let estimated_row = buffer_char_count as u16 / input_width;
        let estimated_col = buffer_char_count as u16 % input_width;

        // Calculate absolute screen coordinates
        let cursor_x = input_area.x + 1 + estimated_col;
        let cursor_y = (input_area.y + 1 + estimated_row)
                           .min(input_area.bottom().saturating_sub(1)); // Clamp Y within box

         // Clamp X within box
         let max_cursor_x = input_area.right().saturating_sub(1);
         let clamped_cursor_x = cursor_x.min(max_cursor_x);

        // Set the terminal cursor
        f.set_cursor(clamped_cursor_x, cursor_y);
    }

    // --- Status Bar ---
    let status_style = if app.status_message.to_lowercase().contains("error") {
        Style::default().bg(Color::Red).fg(Color::White)
    } else if app.is_loading {
        Style::default().bg(Color::Yellow).fg(Color::Black)
    } else {
        Style::default().bg(Color::DarkGray).fg(Color::White)
    };

    // Create status bar paragraph, borrowing from app.status_message
    let status_bar = Paragraph::new(app.status_message.as_str()) // Creates Paragraph<'a>
        .style(status_style);
    f.render_widget(status_bar, right_chunks[2]);
}


// --- Markdown Renderer ---
// Takes a string slice with lifetime 'a and returns Lines borrowing from it
fn render_markdown<'a>(markdown_input: &'a str) -> Vec<Line<'a>> {
    let mut options = Options::empty();
    options.insert(Options::ENABLE_STRIKETHROUGH);
    let parser = Parser::new_ext(markdown_input, options);

    let mut lines: Vec<Line<'a>> = Vec::new();
    let mut current_spans: Vec<Span<'a>> = Vec::new();
    let mut current_style = Style::default().fg(Color::Cyan); // Base style for model text
    let mut list_stack: Vec<Option<u64>> = Vec::new();
    let mut in_code_block = false;
    let code_block_style = Style::default().bg(Color::Rgb(40, 40, 40)).fg(Color::White);
    let inline_code_style = Style::default().bg(Color::Rgb(50, 50, 50)).fg(Color::Yellow).add_modifier(Modifier::ITALIC);

    // Helper closure to push completed lines
    let push_current_line = |lines: &mut Vec<Line<'a>>, current_spans: &mut Vec<Span<'a>>| {
        if !current_spans.is_empty() {
            lines.push(Line::from(current_spans.drain(..).collect::<Vec<_>>()));
        }
    };

    for event in parser {
        match event {
            MDEvent::Start(tag) => {
                match tag {
                    MDTag::Paragraph => {
                        // Reset style for new paragraph if needed (e.g., after blockquote)
                        current_style = Style::default().fg(Color::Cyan);
                    }
                    MDTag::Heading(level, _, _) => {
                        push_current_line(&mut lines, &mut current_spans); // Finish previous line
                        current_style = Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD);
                        let prefix = "#".repeat(level as usize) + " ";
                        current_spans.push(Span::styled(prefix, current_style));
                    }
                    MDTag::BlockQuote => {
                        push_current_line(&mut lines, &mut current_spans);
                        current_style = Style::default().fg(Color::Yellow).add_modifier(Modifier::ITALIC);
                        current_spans.push(Span::styled("> ", current_style));
                    }
                    MDTag::CodeBlock(_) => {
                        push_current_line(&mut lines, &mut current_spans);
                        in_code_block = true;
                        current_style = code_block_style;
                        lines.push(Line::styled("```", Style::default().fg(Color::DarkGray)));
                    }
                    MDTag::List(start_index) => {
                        push_current_line(&mut lines, &mut current_spans); // Ensure previous content is flushed
                        list_stack.push(start_index);
                    }
                    MDTag::Item => {
                        push_current_line(&mut lines, &mut current_spans);
                        let indent = "  ".repeat(list_stack.len().saturating_sub(1));
                        let marker = match list_stack.last() {
                             Some(&Some(start)) => { // Ordered list
                                 // Simple count - may be incorrect for complex lists
                                 let count = lines.iter().filter(|line| {
                                     line.spans.first().map_or(false, |span| span.content.starts_with(&indent) && span.content.contains(". "))
                                 }).count() as u64;
                                 format!("{}. ", start.saturating_add(count))
                             }
                             _ => "* ".to_string(), // Bullet list
                         };
                        current_style = Style::default().fg(Color::Cyan); // Reset item text style
                        current_spans.push(Span::raw(indent)); // Add indentation
                        current_spans.push(Span::styled(marker, Style::default().fg(Color::Green))); // Style marker
                    }
                    MDTag::Emphasis => { // Italics
                        current_style = current_style.add_modifier(Modifier::ITALIC);
                    }
                    MDTag::Strong => { // Bold
                        current_style = current_style.add_modifier(Modifier::BOLD);
                    }
                    MDTag::Strikethrough => {
                        current_style = current_style.add_modifier(Modifier::CROSSED_OUT);
                    }
                    MDTag::Link(_, _dest, _) => { // Mark dest as unused
                        current_style = current_style.fg(Color::Blue).add_modifier(Modifier::UNDERLINED);
                    }
                     MDTag::Image(_, _, _) => { // Placeholder for images
                         current_spans.push(Span::styled("[Image]", Style::default().fg(Color::DarkGray)));
                         // Don't process child events (alt text) for now
                         // This requires more complex state management in the parser loop
                         // Or using a different parsing approach. For now, just the placeholder.
                         // We might need to manually advance the parser past the image content here.
                         // Let's keep it simple and just show placeholder.
                         continue; // Skip processing potential alt text as regular text
                     }
                     // Ignore table tags
                    MDTag::Table(_) | MDTag::TableHead | MDTag::TableRow | MDTag::TableCell => {}
                    // Ignore footnotes
                    MDTag::FootnoteDefinition(_) => {}
                }
            }
            MDEvent::End(tag) => {
                match tag {
                    MDTag::Paragraph | MDTag::Heading(_,_,_) | MDTag::Item | MDTag::BlockQuote => {
                        push_current_line(&mut lines, &mut current_spans);
                        // Reset style might be needed if nested (e.g., end blockquote)
                        // Base style is reset implicitly at start of paragraph usually
                    }
                    MDTag::CodeBlock(_) => {
                        push_current_line(&mut lines, &mut current_spans); // Push last line of code
                        in_code_block = false;
                        lines.push(Line::styled("```", Style::default().fg(Color::DarkGray)));
                        current_style = Style::default().fg(Color::Cyan); // Reset style
                    }
                    MDTag::List(_) => {
                        // If the list ends, ensure any content in the last item is pushed.
                        // This might be handled by the End(Item) already, but check.
                        // push_current_line(&mut lines, &mut current_spans); // Potentially redundant
                        list_stack.pop();
                    }
                    MDTag::Emphasis => {
                        current_style = current_style.remove_modifier(Modifier::ITALIC);
                    }
                    MDTag::Strong => {
                        current_style = current_style.remove_modifier(Modifier::BOLD);
                    }
                    MDTag::Strikethrough => {
                        current_style = current_style.remove_modifier(Modifier::CROSSED_OUT);
                    }
                    MDTag::Link(_, _, _) => {
                        current_style = current_style.remove_modifier(Modifier::UNDERLINED);
                         // Reset color if it was changed for the link
                        if current_style.fg == Some(Color::Blue) {
                           current_style = current_style.fg(Color::Cyan);
                        }
                    }
                    MDTag::Image(_, _, _) => {} // No style changes for image placeholder end
                    // Ignored tags
                    MDTag::Table(_) | MDTag::TableHead | MDTag::TableRow | MDTag::TableCell => {}
                    MDTag::FootnoteDefinition(_) => {}
                }
            }
            MDEvent::Text(text) => {
                // text is Cow<'a, str>
                if in_code_block {
                     // Preserve line breaks within code blocks
                    for (i, code_line) in text.lines().enumerate() {
                        if i > 0 { // Push previous line if multi-line text event
                            push_current_line(&mut lines, &mut current_spans);
                        }
                        current_spans.push(Span::styled(code_line.to_string(), current_style));
                    }
                } else {
                    // Handle potential line breaks in regular text
                     for (i, txt_line) in text.lines().enumerate() {
                          if i > 0 {
                              push_current_line(&mut lines, &mut current_spans);
                          }
                          current_spans.push(Span::styled(txt_line.to_string(), current_style));
                     }
                     // If the text ended with a newline, the last segment was pushed.
                     // If not, it remains in current_spans. Check if we need to push here.
                     if text.ends_with('\n') && !text.is_empty() { // Avoid pushing empty line just for trailing \n
                          push_current_line(&mut lines, &mut current_spans);
                     }
                }
            }
            MDEvent::Code(text) => { // Inline code `code`
                // text is Cow<'a, str>
                current_spans.push(Span::styled(text.to_string(), inline_code_style));
            }
            MDEvent::Html(_) | MDEvent::FootnoteReference(_) => {
                 // Ignored HTML and footnotes
            }
            MDEvent::SoftBreak => {
                // Usually treat as a space in Markdown rendering
                 current_spans.push(Span::raw(" "));
            }
            MDEvent::HardBreak => {
                // Treat as an explicit line break
                 push_current_line(&mut lines, &mut current_spans);
            }
            MDEvent::Rule => {
                // Draw a horizontal rule
                push_current_line(&mut lines, &mut current_spans);
                lines.push(Line::styled("─".repeat(50), Style::default().fg(Color::DarkGray))); // Adjust look
                lines.push(Line::raw("")); // Add spacing after rule
            }
            MDEvent::TaskListMarker(checked) => {
                // Task list item marker like [ ] or [x]
                let marker = if checked { "[x] " } else { "[ ] " };
                current_spans.push(Span::styled(marker, Style::default().fg(Color::Yellow)));
            }
        }
    }

    // Push any remaining spans after the loop finishes
    push_current_line(&mut lines, &mut current_spans);

    // Ensure at least one line is returned, even if empty, for consistent spacing
    if lines.is_empty() {
        lines.push(Line::raw(""));
    }

    lines // Return Vec<Line<'a>>
}


// --- Async Ollama API Functions ---
async fn fetch_models(client: Client) -> Result<Vec<String>, AppError> {
    let url = format!("{}/api/tags", OLLAMA_BASE_URL);
    let response = client.get(&url)
        .timeout(Duration::from_secs(15))
        .send().await
        .map_err(AppError::ApiRequest)?;

    if response.status().is_success() {
        let tags_response: OllamaTagsResponse = response.json().await?;
        Ok(tags_response.models.into_iter().map(|m| m.name).collect())
    } else {
        let status = response.status();
        let err_text = response.text().await.unwrap_or_else(|_| "Failed to read error body".to_string());
        Err(AppError::ApiResponse(format!("Status {} - {}", status, err_text)))
    }
}

async fn stream_ollama_response(
    client: Client,
    model_name: String,
    prompt: String,
    event_sender: mpsc::Sender<AppEvent>,
) -> Result<(), AppError> { // Return AppError for internal task errors (like channel send fail)

    let url = format!("{}/api/generate", OLLAMA_BASE_URL);
    let request_body = OllamaGenerateRequest {
        model: model_name,
        prompt,
        stream: true,
    };

    // Send request and handle potential client-side errors
    let response_result = client.post(&url)
        .json(&request_body)
        .timeout(Duration::from_secs(300)) // Long timeout for generation
        .send().await;

    let response = match response_result {
         Ok(resp) => resp,
         Err(e) => {
             // Report request error via channel
             let err_msg = format!("Request Error: {}", e);
             // Use ?.await notation for cleaner error handling on send
             let _ = event_sender.send(AppEvent::OllamaError(err_msg.clone())).await; // Ignore send error here? Or handle?
             let _ = event_sender.send(AppEvent::OllamaDone).await; // Signal done regardless
             return Err(AppError::ApiRequest(e)); // Propagate original error if needed by caller context (though usually handled via event)
         }
     };

    // Handle API-level errors (non-2xx status codes)
    if !response.status().is_success() {
        let status = response.status();
        let err_text = response.text().await.unwrap_or_else(|_| "Failed to read error body".to_string());
        let err_msg = format!("API Error: Status {} - {}", status, err_text);
        // Send API error and Done signal via channel
        let send_err = event_sender.send(AppEvent::OllamaError(err_msg)).await
             .map_err(|e| AppError::ChannelSend(format!("Failed to send API error: {}", e))); // Convert channel error to AppError
        let send_done = event_sender.send(AppEvent::OllamaDone).await
             .map_err(|e| AppError::ChannelSend(format!("Failed to send done after API error: {}", e)));
        // Return the first channel send error encountered, or Ok if both sends succeeded
        return send_err.and(send_done);
    }

    // Process the successful stream response
    let mut stream = response.bytes_stream();
    let mut buffer = String::new(); // Buffer for partial JSON lines

    while let Some(item_result) = stream.next().await {
        match item_result {
            Ok(chunk_bytes) => {
                // Append bytes to buffer, handling potential UTF-8 errors lossily
                buffer.push_str(&String::from_utf8_lossy(&chunk_bytes));

                // Process complete lines delimited by newline
                while let Some(newline_pos) = buffer.find('\n') {
                    // Extract line and remove from buffer
                    let line = buffer.drain(..=newline_pos).collect::<String>();
                    let trimmed_line = line.trim();

                    if trimmed_line.is_empty() { continue; } // Skip empty lines

                    // Attempt to parse the line as an Ollama chunk
                    match serde_json::from_str::<OllamaGenerateChunk>(trimmed_line) {
                        Ok(chunk) => {
                            // Send the response part via channel
                            if event_sender.send(AppEvent::OllamaChunk(chunk.response)).await.is_err() {
                                eprintln!("Error: Failed to send Ollama chunk to main loop. Stopping stream.");
                                // Report channel error
                                let _ = event_sender.send(AppEvent::OllamaError("Channel closed during streaming".to_string())).await;
                                return Err(AppError::ChannelSend("Failed to send chunk".to_string()));
                            }

                            // Check if this chunk signals the end
                            if chunk.done {
                                // Send the final done signal
                                if event_sender.send(AppEvent::OllamaDone).await.is_err() {
                                     eprintln!("Error: Failed to send Ollama done signal.");
                                     // *** FIX IS HERE *** Changed AppEvent to AppError
                                     return Err(AppError::ChannelSend("Failed to send done signal".to_string()));
                                }
                                return Ok(()); // Stream finished successfully
                            }
                        }
                        Err(e) => {
                            // Report JSON decoding errors
                            let error_msg = format!("JSON Decode Error: '{}' on line: '{}'", e, trimmed_line);
                            eprintln!("{}", error_msg); // Log locally
                            // Send error via channel but continue processing stream
                            if event_sender.send(AppEvent::OllamaError(error_msg)).await.is_err() {
                                eprintln!("Error: Failed to send JSON decode error to main loop.");
                                // If sending error fails, the channel is likely broken, stop the task
                                return Err(AppError::ChannelSend("Failed to send decode error".to_string()));
                            }
                        }
                    }
                } // End while processing lines
            }
            Err(e) => {
                 // Error reading from the byte stream itself
                 let error_msg = format!("Stream Read Error: {}", e);
                 eprintln!("{}", error_msg);
                 // Report error via channel
                 if event_sender.send(AppEvent::OllamaError(error_msg)).await.is_err() {
                      eprintln!("Error: Failed to send stream read error to main loop.");
                      // Even if send fails, proceed to send Done signal if possible
                 }
                 // Signal done because the stream stopped
                 let _ = event_sender.send(AppEvent::OllamaDone).await;
                 // Return Ok as the error was reported via channel. Task itself didn't fail due to channel send.
                 // Or could return a specific AppError indicating stream read failure.
                 return Ok(());
            }
        }
    } // End while reading stream chunks

    // --- Stream Ended Naturally ---

     // Process any remaining data in the buffer after the stream closes
    let final_data = buffer.trim();
    if !final_data.is_empty() {
         match serde_json::from_str::<OllamaGenerateChunk>(final_data) {
             Ok(chunk) => {
                 // Send final chunk content
                 if event_sender.send(AppEvent::OllamaChunk(chunk.response)).await.is_err() {
                      eprintln!("Error: Failed to send final Ollama chunk to main loop.");
                      // Proceed to send Done signal anyway
                 }
                 // Check chunk.done here? Maybe, but stream ending is the main signal.
             }
             Err(e) => {
                 // Report final buffer decoding error
                 let error_msg = format!("Final Buffer Decode Error: '{}' on data: '{}'", e, final_data);
                 eprintln!("{}", error_msg);
                  if event_sender.send(AppEvent::OllamaError(error_msg)).await.is_err(){
                       eprintln!("Error: Failed to send final buffer decode error to main loop.");
                  }
             }
         }
    }

    // Always send the Done signal when the stream ends, unless already sent (e.g., chunk.done was true)
    // Note: The logic above returns early if chunk.done is true and Done is sent successfully.
    if event_sender.send(AppEvent::OllamaDone).await.is_err() {
        eprintln!("Error: Failed to send final Ollama done signal to main loop.");
        return Err(AppError::ChannelSend("Failed to send final done signal".to_string()));
    }

    Ok(()) // Task completed successfully
}
