# Import necessary modules from Django
from django.contrib.auth import logout
from django.shortcuts import redirect, render
from django.views.generic import ListView, FormView
from datetime import timedelta
# Import check_availability function from a module

from django.utils import timezone
# Create your models here.
def homepage(request):
    return render(request, 'index.html')