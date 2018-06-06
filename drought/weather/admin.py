from django.contrib import admin
from .models import WeatherModel

# Register your models here.
@admin.register(WeatherModel)
class WeatherModelAdmin(admin.ModelAdmin):
	list_display = ('date', 'area')
