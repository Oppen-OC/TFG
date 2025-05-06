from django.db import models

class Licitacion(models.Model):
    cod = models.TextField()
    id_publicacion_ted = models.TextField(blank=True)
    organo_contratacion = models.TextField()
    id_organo_contratacion = models.TextField()
    estado_licitacion = models.TextField()
    objeto_contrato = models.TextField()
    financiacion_ue = models.TextField()
    presupuesto_base_licitacion = models.TextField()
    valor_estimado_contrato = models.TextField()
    tipo_contrato = models.TextField()
    codigo_cpv = models.TextField()
    lugar_ejecucion = models.TextField()
    sistema_contratacion = models.TextField()
    procedimiento_contratacion = models.TextField()
    tipo_tramitacion = models.TextField()
    metodo_presentacion_oferta = models.TextField()
    resultado = models.TextField()
    adjudicatario = models.TextField()
    num_licitadores_presentados = models.TextField()
    importe_adjudicacion = models.TextField()    
    
    def __str__(self):
        return self.cod

class Codigos(models.Model):
    cod = models.TextField()
    url = models.TextField()

    def __str__(self):
        return self.cod

