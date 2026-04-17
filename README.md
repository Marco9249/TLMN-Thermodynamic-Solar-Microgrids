# 🌍 Thermodynamic Liquid Manifold Networks (TLMN)

## 📖 نظرة عامة
يحتوي هذا المسار على الكود المصدري وبيانات التدريب الخاصة بالورقة العلمية "Thermodynamic Liquid Manifold Networks: Physics-Bounded Deep Learning for Solar Forecasting in Autonomous Off-Grid Microgrids".

هذا المشروع يعتمد على معمارية عميقة، وللمقارنة المعمارية والاختبارات تم تحليل وضعيات تتمركز فيها طبقة الانتباه (Attention Layer) مباشرة بعد طبقة BiLSTM. كما يتم الاعتماد بشكل أساسي على البيانات المستخرجة من نظام متغيرات NASA POWER، مع تطبيق نافذة بيانات منزلقة بمقدار 3 خطوات زمنية (Sliding Window of 3 Time Steps) لضمان المعالجة الزمنية المثلى.

## 📐 معمارية النموذج (v3)
- التخلي عن `LiquidNeuralODE` واستبداله بـ `1D-CNN Temporal Encoder` لتجنب التباطؤ الزمني.
- البوابات الفيزيائية `PhysicsGatedOutput` لضمان عدم وجود إنتاج وهمي للإشعاع ليلاً.
- استخدام دالة `Log-Cosh Loss`.

## 🛠 التشغيل
1. تثبيت الحزم من `requirements.txt`.
2. تشغيل كود `TLMN_Model.py`.

## 🔖 الاقتباس المرجعي (APA 7th Edition)
Abdullah, M. E. B. (2026). *Thermodynamic Liquid Manifold Networks: Physics-Bounded Deep Learning for Solar Forecasting in Autonomous Off-Grid Microgrids*. arXiv. https://arxiv.org/abs/2604.11909

---
© محفوظة رسميًا للباحث المهندس / م. محمد عزالدين بابكر عبدالله - 2026
