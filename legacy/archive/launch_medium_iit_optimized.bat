@echo off
echo 🚀 LANZANDO ENTRENAMIENTO MEDIUM IIT OPTIMIZADO
echo =============================================
echo Configuracion: lambda_phi=0.01, dropout=0.25
echo Arquitectura: medium_iit (65.3M parametros)
echo Mejora esperada: 96.8%%
echo.

python train_v5_2_wikitext_real.py ^
    --model-size medium_iit ^
    --lambda-phi 0.01 ^
    --dropout 0.25 ^
    --learning-rate 0.0002 ^
    --batch-size 8 ^
    --epochs 3 ^
    --gradient-accumulation 4

echo.
echo ✅ Entrenamiento medium_iit completado
pause
