from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from fontTools.misc.eexec import encrypt, decrypt

from .forms import UserRegisterForm, UserUpdateForm, ProfileUpdateForm
from django.contrib.auth.decorators import login_required
import smtplib
from django.contrib.auth.models import User
import base64
from django.contrib.auth.hashers import make_password, check_password


# Create your views here.

def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            print("password =", password)
            messages.success(request, 'Your account has been created! You can now login.')
            return redirect('login')
    else:
        form = UserRegisterForm()
    return render(request, 'users/register.html', {'form': form})


@login_required()
def profile(request):
    if request.method == 'POST':
        u_form = UserUpdateForm(request.POST, request.FILES, instance=request.user)
        p_form = ProfileUpdateForm(request.POST, request.FILES, instance=request.user.profile)
        if u_form.is_valid() and p_form.is_valid():
            u_form.save()
            p_form.save()
            messages.success(request, 'Your account has been updated!')
            return redirect('profile')
    else:
        u_form = UserUpdateForm(instance=request.user)
        p_form = ProfileUpdateForm(instance=request.user.profile)

    context = {
        'u_form': u_form,
        'p_form': p_form,
    }

    return render(request, 'users/profile.html', context)


@login_required()
def logout(request):
    request.session.flush()
    return redirect('/login')


#
# def change_password(request):
#     specific_user = User.objects.get(email='likitha.aluru2002@gmail.com')
#     print(specific_user.username)
#     print(specific_user.email)
#     hashed_password = specific_user.password
#     print(hashed_password)
#     # Decode the hashed password using base64
#     decoded_password = base64.b64decode(hashed_password.decode("utf-8"))
#     print(decoded_password)
#     return render(request, "users/change_password.html", {})

# def customer_change_password(request):
#     email = request.session["email"]
#     if customer_is_login(request):
#         if request.method == "POST":
#             password = encrypt(request.POST['password'])
#             print(password)
#             pas = decrypt(password)
#             print(pas)
#             new_password = request.POST["new_password"]
#             print(new_password)
#             try:
#                 print("hii")
#                 user = Customer.objects.get(email=email)
#                 pwd = user.password
#                 opwd = decrypt(pwd)
#                 print(opwd)
#                 print("hello")
#                 if pas == opwd:
#                     encryptpass = encrypt(request.POST['new_password'])
#                     print(encryptpass)
#                     user.password = encryptpass
#                     print("hello2")
#                     user.save()
#                     return render(request, "customer_login.html", {"msg": "Password Update Successful"})
#                 else:
#                     return render(request, "customer_change_password.html",
#                                   {"msg": "Old Password Is Wrong", "email": email})
#             except Exception as e:
#                 print(e)
#                 return render(request, "customer_change_password.html", {"msg": "Invalid Data", "email": email})
#         else:
#             return render(request, "customer_change_password.html", {"email": email})


def change_password(request):
    if request.method == 'POST':
        old_password = request.POST.get('old_password')
        print("old_password = ", old_password)
        new_password = request.POST.get('new_password')
        print("new_password = ", new_password)
        # Retrieve the current user
        current_user = request.user
        # Verify old password
        if check_password(old_password, current_user.password):
            # Hash the new pass
            hashed_new_password = make_password(new_password)
            # Update the user's password
            current_user.password = hashed_new_password
            current_user.save()
            messages.success(request, 'Password changed Successfully.')
            return redirect('login')
        else:
            messages.error(request, 'Invalid old Password.')
            return redirect('change_password')
    return render(request, "users/change_password.html", {})
