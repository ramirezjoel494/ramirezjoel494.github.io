import requests
from bs4 import BeautifulSoup
import pandas as pd
import speech_recognition as sr
import pyttsx3
import time
from io import StringIO


def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    time.sleep(0.5)

def listen_command():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Escuchando...")
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source, timeout=5, phrase_time_limit=7)
        command = r.recognize_google(audio, language='es-ES')
        print(f"Reconocido: {command}\n")
        return command.lower()
    except Exception as e:
        print(f"Error o no hay micrófono: {e}")
        # Si falla, pedir entrada por texto
        command = input("Escribe tu respuesta: ")
        return command.lower()

def obtener_estadisticas_lvbp(season_id=2024, team_id=695, game_type='R', stat_type='hittings', tab_id=None):
    url = 'https://stats.lvbp.com/resources/views/team_stats.php'
    data = {
        'season_id': season_id,
        'team_id': team_id,
        'game_type': game_type,
        'type': stat_type
    }
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Referer': 'https://stats.lvbp.com/equipos.php',
        'X-Requested-With': 'XMLHttpRequest'
    }
    response = requests.post(url, data=data, headers=headers)
    if response.status_code != 200:
        speak(f"Error en la petición: código {response.status_code}")
        return None
    soup = BeautifulSoup(response.text, 'html.parser')
    if tab_id:
        div_tab = soup.find('div', id=tab_id)
        if not div_tab:
            speak(f"No se encontró la pestaña con id {tab_id}")
            return None
        tabla = div_tab.find('table', class_='table table-sm table-hover table-stats table-striped table-bordered')
    else:
        tabla = soup.find('table', class_='table table-sm table-hover table-stats table-striped table-bordered')
    if not tabla:
        speak("No se encontró la tabla con las estadísticas.")
        return None
    df = pd.read_html(StringIO(str(tabla)))[0]
    df.columns = [col.strip() for col in df.columns]
    return df

def preguntar_temporada():
    speak("¿De qué temporada quieres las estadísticas? Por ejemplo, 2024.")
    respuesta = listen_command()
    try:
        anio = int(respuesta)
        return anio
    except:
        speak("No entendí el año, usaré 2024 por defecto.")
        return 2024

def preguntar_equipo():
    equipos = {
        "leones": 695,
        "tiburones": 698,
        "cardenales": 693,
        "navegantes": 696,
        "águilas": 692,
        "caribes": 694,
        "bravos": 697,
        "tigres": 699
    }
    speak("¿De qué equipo quieres las estadísticas? Puedes decir Leones, Tiburones, Cardenales, Navegantes, Águilas, Caribes, Bravos o Tigres.")
    respuesta = listen_command()
    for nombre, id_equipo in equipos.items():
        if nombre in respuesta:
            return id_equipo, nombre
    speak("No reconocí el equipo, usaré Leones por defecto.")
    return 695, "leones"

def preguntar_clasificacion():
    speak("¿Qué tipo de clasificación quieres? Puedes decir Regular, Playoffs, Semifinal o Final.")
    respuesta = listen_command()
    if respuesta:
        if "regular" in respuesta:
            return 'R'
        elif "playoffs" in respuesta:
            return 'D'
        elif "semifinal" in respuesta:
            return 'L'
        elif "final" in respuesta:
            return 'W'
    speak("No entendí el tipo de clasificación, usaré Regular por defecto.")
    return 'R'


def preguntar_tipo_estadistica():
    speak("¿Qué tipo de estadísticas quieres? Puedes decir bateo, pitcheo o defensa.")
    respuesta = listen_command()
    if "bateo" in respuesta or "bateador" in respuesta:
        return 'hittings'
    elif "pitcheo" in respuesta or "lanzador" in respuesta:
        return 'pitchings'
    elif "defensa" in respuesta:
        return 'fieldings'
    else:
        speak("No entendí el tipo, usaré bateo por defecto.")
        return 'hittings'

def preguntar_avanzadas():
    speak("¿Quieres estadísticas avanzadas? Di sí o no.")
    respuesta = listen_command()
    if respuesta and ("sí" in respuesta or "si" in respuesta):
        return 'adv_bat-tab-pane'
    else:
        return None


def resumen_jugadores(df, stat_column='PEB'):
    if stat_column not in df.columns:
        speak(f"No se encontró la estadística {stat_column} para hacer el resumen.")
        return

    # Excluir filas resumen o totales
    df_clean = df[~df['JUGADOR'].str.lower().str.contains('total|totales|relevistas|promedio|equipo', na=False)].copy()

    # Convertir a numérico y eliminar filas con NaN en la columna estadística
    df_clean[stat_column] = pd.to_numeric(df_clean[stat_column], errors='coerce')
    df_clean = df_clean.dropna(subset=[stat_column])

    if df_clean.empty:
        speak("No hay datos válidos para hacer el resumen.")
        return

    mejor = df_clean.loc[df_clean[stat_column].idxmax()]
    peor = df_clean.loc[df_clean[stat_column].idxmin()]

    resumen = (f"\n\nEl mejor jugador según, {stat_column} es, {mejor['JUGADOR']}, con un valor de {mejor[stat_column]:.3f}. "
               f"\n\nEl peor jugador es, {peor['JUGADOR']}, con un valor de {peor[stat_column]:.3f}.")
    print(resumen)
    speak(resumen)



def main():
    speak("Hola, este algoritmo se usa para consultar estadísticas de los equipos de la LVBP.")
    season_id = preguntar_temporada()
    team_id, nombre_equipo = preguntar_equipo()
    stat_type = preguntar_tipo_estadistica()
    tab_id = preguntar_avanzadas()
    game_type = preguntar_clasificacion()

    speak(f"Obteniendo estadísticas para la temporada {season_id}, equipo {nombre_equipo}, tipo {stat_type}, clasificación {game_type}.")
    df = obtener_estadisticas_lvbp(season_id, team_id, game_type, stat_type, tab_id)
    df_2 = df
    df_2.to_csv(f'{season_id}_{nombre_equipo}_{stat_type}_{game_type}.csv', index=False)
    if df is not None:
        speak("Datos obtenidos correctamente. Tambien almacené el archivo csv para visualización con tableu. Te mostraré las primeras filas y los destacados.")

        print("Primeras filas del dataframe: ")
        print("\n\n",df.head())

        if stat_type == 'hittings':
            columna_resumen = 'PEB'
        elif stat_type == 'pitchings':
            # Ejemplo: usar 'ERA' (promedio de carreras limpias) para pitcheo, si existe
            columna_resumen = 'ERA' if 'ERA' in df.columns else df.columns[1]
        elif stat_type == 'fieldings':
            # Ejemplo: usar 'PORC' (porcentaje de fildeo) para defensa, si existe
            columna_resumen = 'PORC' if 'PORC' in df.columns else df.columns[1]
        else:
            columna_resumen = df.columns[1]

        resumen_jugadores(df, columna_resumen)

    else:
        speak("No se pudo obtener la información solicitada.")

if __name__ == "__main__":
    main()



