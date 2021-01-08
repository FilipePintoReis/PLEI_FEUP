'''
This module is responsible for emails
'''
import sys
import smtplib

from email_pkg.credentials import address, password

class EmailSender:
    @staticmethod
    def start_server():
        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()

            server.login(address, password)
            print("Login success")
        except Exception:
            print("Could not start smtp server:", sys.exc_info()[0])

        return server

    @staticmethod
    def send_email(server, receiver, body):
        try:
            server.sendmail(address, receiver, body)
        except Exception:
            print("Could not send email:", sys.exc_info()[0])
            with open('log.txt', 'a+') as file:
                file.write(f'\n\n\n {body}')

        print('email sent')
