from django.db import models
import pandas as pd
from django.utils import timezone
# Create your models here.

class WeatherModel(models.Model):
	date = models.DateField(null=True)
	area = models.CharField(max_length=30, null=True)
