"""
Function Decoration class to check for credentials
"""
from flask import redirect, flash, session, url_for, request
from functools import wraps


def home_logged_in(func):
    """
    Function: home_logged_in
    Input: function
    Returns: Decorator to check if session is logged in when accessing home
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'logged_in' in session:
            return func(*args, **kwargs)
        else:
            return redirect(url_for('login'))
    return wrapper


def check_logged_in(func):
    """
    Function: check_logged_in
    Input: function
    Returns: Decorator to check if session is logged in
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'logged_in' in session:
            return func(*args, **kwargs)
        else:
            flash("You need to login first.", "danger")
            return redirect(url_for('login'))
    return wrapper


def wrong_info(func):
    """
    Function: wrong_info
    Input: function
    Returns: Decorator to check if user entered right credentials
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if (request.form['username'] != 'admin' or request.form['password'] != 'password'):
            flash("You did not enter the right credentials.", "danger")
            return redirect(url_for('login'))
        else:
            flash("You successfully logged in!", "success")
            return func(*args, **kwargs)
    return wrapper
