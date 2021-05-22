from flask import Flask, redirect, flash, session, url_for, request
from functools import wraps

def home_logged_in(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'logged_in' in session:
            return func(*args, **kwargs)
        else:
          return redirect(url_for('login'))
    return wrapper


def check_logged_in(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'logged_in' in session:
            return func(*args, **kwargs)
        else:
          flash('You need to login first.'.format(request.form.get("title")), "danger")
          return redirect(url_for('login'))
    return wrapper


def wrong_info(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if (request.form['username'] != 'admin' or request.form['password'] != 'password'):
            flash('You did not enter the right credentials.'.format(request.form.get("title")), "danger")
            return redirect(url_for('login'))
        else:
            flash('You successfully logged in!'.format(request.form.get("title")), "success")
            return func(*args, **kwargs)
    return wrapper
