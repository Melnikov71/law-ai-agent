from main import IncomingDocument, Products
from main import AgreementGenerator

from datetime import date
import os


def test_incoming_document_creation():
    """Проверка создания документа"""
    document = IncomingDocument(doc_number='№ 005-04/9702/20', doc_date=date(2024, 9, 10))
    assert document.doc_number == '№ 005-04/9702/20'
    assert document.doc_date == date(2024, 9, 10)


def test_products_creation():
    """Проверка создания продукта"""
    product = Products(okpd='25', placement='Дальневосточный')
    assert product.okpd == '25'
    assert product.placement == 'Дальневосточный'


def test_sanitize_doc_number():
    """Проверка функции sanitize_doc_number()"""
    document = IncomingDocument(doc_number='№ 005-04/9702/20', doc_date=date(2024, 9, 10))
    products = [Products(okpd='25', placement='Дальневосточный'), Products(okpd='10', placement='Сибирский')]
    sanitize_doc_number = AgreementGenerator(document, products)._sanitize_doc_number(document.doc_number)
    assert sanitize_doc_number == '005-04/9702/20'


def test_sanitize_filename():
    """Проверка функции sanitize_doc_number()"""
    document = IncomingDocument(doc_number='№ 005-04/9702/20', doc_date=date(2024, 9, 10))
    products = [Products(okpd='25', placement='Дальневосточный'), Products(okpd='10', placement='Сибирский')]
    generator = AgreementGenerator(document, products)
    sanitize_doc_number = generator._sanitize_doc_number(document.doc_number)
    output_file_path = generator._sanitize_filename(
        f'Согласие на участие в аукционе по приглашению № {sanitize_doc_number} от {document.doc_date.strftime("%d.%m.%Y")}.docx')
    assert output_file_path == 'Согласие на участие в аукционе по приглашению № 005-04_9702_20 от 10.09.2024.docx'


def test_agreement_creation():
    """Проверка создания выходного файла согласия"""
    document = IncomingDocument(doc_number='№ 005-04/9702/20', doc_date=date(2024, 9, 10))
    products = [Products(okpd='25', placement='Дальневосточный'), Products(okpd='10', placement='Сибирский')]

    generator = AgreementGenerator(document, products)
    generator.generate()

    sanitize_doc_number = generator._sanitize_doc_number(document.doc_number)

    output_file_path = generator._sanitize_filename(
        f'Согласие на участие в аукционе по приглашению № {sanitize_doc_number} от {document.doc_date.strftime("%d.%m.%Y")}.docx')

    assert os.path.exists(output_file_path)
    os.remove(output_file_path)
