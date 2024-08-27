# TC2008B. Sistemas Multiagentes y Gráficas Computacionales
# Python server to interact with Unity
# Sergio. Julio 2021
# Actualización Lorena Martínez Agosto 2021

from http.server import BaseHTTPRequestHandler, HTTPServer
import logging
import json

from agents_test import *
# Importa tu lógica de agentes aqui:
def miLogica():
    model = Model(parameters)
    run = model.run()

    info = []
    for agent in model.pedestrianAgents:
        posi = {"id" : agent.id, "positions": agent.vectors}
        info.append(posi)
   
    # Send parametrs, send result stored in postion
    json_output = {"players" : info}
    json_output = json.dumps(json_output)

    json_parameters = {"parameters" : parameters }
    json_parameters = json.dumps(json_parameters)

    data1 = json.loads(json_output)
    data2 = json.loads(json_parameters)

    merge = {**data2, **data1}
    merge_json = json.dumps(merge)

    return merge_json

def miLogica2():
    model = Model(parameters)
    run = model.run()

    # Guarda los vectores de los agentes en un archivo
    with open("agent_vectors.txt", "w") as file:
        for agent in model.pedestrianAgents:
            for vector in agent.vectors:
                v1 = str(vector[0])
                v2 = str(vector[1])
                file.write(f"{v1}, {v2} \n")

    return "Agent vectors saved in agent_vectors.txt"


#Esta función convierte a json una secuencia
def positionsToJSON(ps):
    posDICT = []
    for p in ps:
        pos = {
            "x" : p[0],
            "z" : p[1],
            "y" : p[2]
        }
        posDICT.append(pos)
    return json.dumps(posDICT)

#El rey del server:
class Server(BaseHTTPRequestHandler):
    
    def _set_response(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
    def do_GET(self):
        logging.info("GET request,\nPath: %s\nHeaders:\n%s\n", str(self.path), str(self.headers))
        self._set_response()
        self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))

        if self.path == '/run-logic':  # Specify the path to trigger miLogica
            result = miLogica()  # Call your function
            self._set_response()
            self.wfile.write(result.encode('utf-8')) 
        elif self.path == '/run-logic2':  # Specify the path to trigger miLogica2
            result = miLogica2()  # Call your function
            self._set_response()
            self.wfile.write(result.encode('utf-8'))
        else:
            self._set_response()
            self.wfile.write("GET request for {}".format(self.path).encode('utf-8'))

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        #post_data = self.rfile.read(content_length)
        post_data = json.loads(self.rfile.read(content_length))
        #logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
                     #str(self.path), str(self.headers), post_data.decode('utf-8'))
        logging.info("POST request,\nPath: %s\nHeaders:\n%s\n\nBody:\n%s\n",
                     str(self.path), str(self.headers), json.dumps(post_data))
        
        # AQUI ACTUALIZA LO QUE SE TENGA QUE ACTUALIZAR
        self._set_response()
        #AQUI SE MANDA EL SHOW
        resp = "{\"data\":" + positionsToJSON(positions) + "}"
        #print(resp)
        self.wfile.write(resp.encode('utf-8'))


def run(server_class=HTTPServer, handler_class=Server, port=8585):
    logging.basicConfig(level=logging.INFO)
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    logging.info("Starting httpd...\n") # HTTPD is HTTP Daemon!
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:   # CTRL+C stops the server
        pass
    httpd.server_close()
    logging.info("Stopping httpd...\n")

if __name__ == '__main__':
    from sys import argv
    
    if len(argv) == 2:
        run(port=int(argv[1]))
    else:
        run()
