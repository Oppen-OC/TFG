from django.db import models

class Licitacion(models.Model):
    cod = models.TextField()
    id_publicacion_ted = models.TextField(null=True, blank=True)
    organo_contratacion = models.TextField(null=True, blank=True)
    id_organo_contratacion = models.TextField(null=True, blank=True)
    estado_licitacion = models.TextField(null=True, blank=True)
    objeto_contrato = models.TextField(null=True, blank=True)
    financiacion_ue = models.TextField(null=True, blank=True)
    presupuesto_base_licitacion = models.TextField(null=True, blank=True)
    valor_estimado_contrato = models.TextField(null=True, blank=True)
    tipo_contrato = models.TextField(null=True, blank=True)
    codigo_cpv = models.TextField(null=True, blank=True)
    lugar_ejecucion = models.TextField(null=True, blank=True)
    sistema_contratacion = models.TextField(null=True, blank=True)
    procedimiento_contratacion = models.TextField(null=True, blank=True)
    tipo_tramitacion = models.TextField(null=True, blank=True)
    metodo_presentacion_oferta = models.TextField(null=True, blank=True)
    resultado = models.TextField(null=True, blank=True)
    adjudicatario = models.TextField(null=True, blank=True)
    num_licitadores_presentados = models.TextField(null=True, blank=True)
    importe_adjudicacion = models.TextField(null=True, blank=True)
    precio_unit = models.TextField(null=True, blank=True)
    tanto_alzado = models.TextField(null=True, blank=True)
    tanto_alzado_con = models.TextField(null=True, blank=True)
    criterio_juicio_val = models.TextField(null=True, blank=True)
    evaluables_autom = models.TextField(null=True, blank=True)
    criterio_ecónomico = models.TextField(null=True, blank=True)
    memoria_constructiva = models.TextField(null=True, blank=True)
    programa_trabajos = models.TextField(null=True, blank=True)
    analisis_proyecto = models.TextField(null=True, blank=True)

    def __str__(self):
        return self.cod

class Codigos(models.Model):
    cod = models.TextField()
    url = models.TextField()

    def __str__(self):
        return self.cod

