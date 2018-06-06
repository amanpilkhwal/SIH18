from rest_framework import serializers
from .models import WeatherModel

class WeatherModelSerializer(serializers.ModelSerializer):
 	class Meta:
 		model = WeatherModel
 		fields = '__all__'
 		
 			