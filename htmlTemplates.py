css = '''
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">

<style>
body {
    background-color: #1e1e2f;
    color: #dcdcdc;
    font-family: Arial, sans-serif;
}

.chat-container {
    max-width: 800px;
    margin: 0 auto;
    padding: 1rem;
}

.chat-message {
    padding: 1.25rem;
    border-radius: 10px;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
}

.chat-message.user {
    background-color: #2c2f3f;
}

.chat-message.bot {
    background-color: #3a3f54;
}

.chat-message .avatar {
    font-size: 1.5rem;
    width: 50px;
    height: 50px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-right: 1rem;
    color: #fff;
}

.chat-message .avatar .fa-user {
    color: #6c63ff;
}

.chat-message .avatar .fa-robot {
    color: #ffa500;
}

.chat-message .message {
    flex-grow: 1;
    color: #e0e0e0;
    font-size: 1rem;
}

input[type="text"] {
    background-color: #2b2b3a;
    color: #dcdcdc;
    border: 1px solid #444;
    padding: 0.75rem;
    border-radius: 8px;
    width: 100%;
    font-size: 1rem;
}

input[type="text"]:focus {
    outline: none;
    border-color: #6c63ff;
}

button {
    background-color: #6c63ff;
    color: #fff;
    padding: 0.75rem 1.25rem;
    border: none;
    border-radius: 8px;
    cursor: pointer;
    font-size: 1rem;
    margin-top: 1rem;
}

button:hover {
    background-color: #5a52e0;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <i class="fas fa-robot"></i>
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <i class="fas fa-user"></i>
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
