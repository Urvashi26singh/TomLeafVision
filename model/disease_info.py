def get_disease_info(disease_name, confidence=0.0):
    """
    Get detailed information about a tomato disease
    """
    disease_key = disease_name.lower().strip()
    
    disease_db = {
        'early_blight': {
            'disease': 'Early Blight',
            'description': 'Early blight is a common tomato disease caused by the fungus Alternaria solani. It appears as dark spots with concentric rings on lower leaves, eventually causing yellowing and leaf drop.',
            'symptoms': '• Dark brown spots with concentric rings (target-like appearance)\n• Yellowing around spots\n• Leaves dying from bottom up\n• Stem lesions',
            'treatment': '• Remove infected leaves immediately\n• Apply copper-based fungicides every 7-10 days\n• Use organic options like neem oil\n• Ensure proper air circulation',
            'prevention': '• Mulch around plants to prevent soil splash\n• Water at base of plants, not overhead\n• Rotate crops yearly\n• Plant resistant varieties\n• Maintain proper plant spacing',
            'severity': 'High'
        },
        'late_blight': {
            'disease': 'Late Blight',
            'description': 'Late blight is a devastating disease caused by the water mold Phytophthora infestans. It spreads rapidly in cool, wet conditions and can destroy entire crops within days.',
            'symptoms': '• Large, dark, greasy-looking spots on leaves\n• White fuzzy growth on undersides in humid conditions\n• Dark brown lesions on stems\n• Brown rotting spots on fruits',
            'treatment': '• Remove and destroy infected plants immediately\n• Apply copper-based fungicides preventatively\n• Use fungicides containing chlorothalonil\n• Remove volunteer plants',
            'prevention': '• Ensure good air circulation\n• Avoid overhead watering\n• Space plants properly\n• Plant resistant varieties\n• Monitor weather conditions',
            'severity': 'Very High'
        },
        'healthy': {
            'disease': 'Healthy',
            'description': 'Your tomato plant appears healthy with no signs of disease. Continue good care practices to maintain plant health.',
            'symptoms': '• No visible disease symptoms\n• Green, vigorous leaves\n• Healthy stems\n• Normal growth pattern\n• Good fruit development',
            'treatment': '• No treatment needed\n• Continue regular care\n• Monitor for any changes',
            'prevention': '• Maintain regular watering schedule\n• Provide adequate nutrients\n• Monitor for pests\n• Ensure proper spacing\n• Practice crop rotation',
            'severity': 'None'
        }
    }
    
    if disease_key in disease_db:
        info = disease_db[disease_key].copy()
    else:
        info = {
            'disease': disease_name.replace('_', ' ').title(),
            'description': f'Information for {disease_name} is being updated.',
            'symptoms': 'Please consult a plant specialist for accurate diagnosis.',
            'treatment': '• Remove affected plant parts\n• Monitor plant health\n• Consult local extension service',
            'prevention': '• Maintain good garden hygiene\n• Proper watering practices\n• Regular monitoring',
            'severity': 'Unknown'
        }
    
    info['confidence'] = round(confidence * 100, 2)
    return info