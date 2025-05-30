from dataclasses import dataclass, asdict
from datetime import date
import os
from pathlib import Path
import re
from typing import Sequence
from uuid import uuid4

from dotenv import find_dotenv, load_dotenv
import img2pdf
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langchain_gigachat.chat_models import GigaChat
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from mailmerge import MailMerge

load_dotenv(find_dotenv())

INCOMING_FILE = "Приглашение.jpg"
AGREEMENT_TEMPLATE = "agreement_template.docx"


@dataclass
class IncomingDocument:
    doc_number: str
    doc_date: date


@dataclass
class Products:
    okpd: str
    placement: str


class AgreementGenerator:
    """Класс генерации документа-согласия на участие в аукционе"""

    def __init__(self, incoming_document_data: IncomingDocument, products: list[Products]):
        self.template = AGREEMENT_TEMPLATE
        if not os.path.exists(self.template):
            raise FileNotFoundError(f"Шаблон '{self.template}' не найден в текущей директории.")

        self.incoming_document_data = incoming_document_data
        self.products = [asdict(product) for product in products]

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """
        Удаляет недопустимые символы из имени файла.
        """
        # Заменяем недопустимые символы на "_"
        return re.sub(r'[<>:"/\\|?*]', '_', filename)

    @staticmethod
    def _sanitize_doc_number(doc_number: str) -> str:
        """
        Удаляет недопустимые символы из номера документа.
        """
        return re.sub(r'[№Nn\s]', '', doc_number)

    def generate(self):
        """Генерирует согласие на участие в аукционе на основе данных входящего документа"""

        # Открываем шаблон
        document = MailMerge(self.template)

        # Заменяем поля данными входящего номера и даты в тексте шаблона
        document.merge(
            incoming_number=self._sanitize_doc_number(self.incoming_document_data.doc_number),
            incoming_date=self.incoming_document_data.doc_date.strftime("%d.%m.%Y"),
        )

        # Заполняем таблицу с продукцией
        document.merge_rows("okpd", self.products)

        # Сохраняем заполненное согласие на участие в аукционе
        self.output_document = f'Согласие на участие в аукционе по приглашению № {self._sanitize_doc_number(self.incoming_document_data.doc_number)} от {self.incoming_document_data.doc_date.strftime("%d.%m.%Y")}.docx'
        document.write(self._sanitize_filename(self.output_document))


class LLMAgent:
    def __init__(self, model: LanguageModelLike, tools: Sequence[BaseTool]):
        self._model = model
        self._agent = create_react_agent(
            model,
            tools=tools,
            checkpointer=InMemorySaver())
        self._config: RunnableConfig = {
            "configurable": {"thread_id": uuid4().hex}}

    def upload_file(self, file):
        file_uploaded_id = self._model.upload_file(file).id_  # type: ignore
        return file_uploaded_id

    def invoke(
            self,
            content: str,
            attachments: list[str] | None = None,
            temperature: float = 1.0
    ) -> str:
        """Отправляет сообщение в чат"""
        message: dict = {
            "role": "user",
            "content": content,
            **({"attachments": attachments} if attachments else {})
        }
        return self._agent.invoke(
            {
                "messages": [message],
                "temperature": temperature
            },
            config=self._config)["messages"][-1].content


@tool
def generate_request_document(incoming_document: IncomingDocument, products: list[Products]) -> None:
    """Генерирует документ-заявку на участие в аукционе
    Arg:
        incoming_document: Номер и дата документа, на основании которого создаётся заявка
        products: Список продукции
    """
    AgreementGenerator(incoming_document, products).generate()


def print_agent_response(llm_response: str) -> None:
    print(f"\033[35m{llm_response}\033[0m")


def get_user_prompt() -> str:
    return input("\nТы: ")


def main():
    model = GigaChat(
        model="GigaChat-2-Max",
        verify_ssl_certs=False,
    )

    agent = LLMAgent(model, tools=[generate_request_document])
    system_prompt = ("Твоя задача извлечь исходящий номер, дату документа и список видов продукции с полями 'Класс "
                     "ОКПД' и 'Территориальное размещение (федеральный округ)' из предоставленного документа и на "
                     "основании этих данных сформировать документ-согласие на участие в аукционе."
                     "Не придумывай никаких данных, всё необходимое возьми из предоставленного документа.")

    output_file = INCOMING_FILE

    if Path(INCOMING_FILE).suffix.lower() in [".jpg", ".jpeg"]:
        # Конвертируем изображение в PDF
        output_file = Path(INCOMING_FILE).with_suffix(".pdf")
        with open(output_file, "wb") as f:
            f.write(img2pdf.convert(INCOMING_FILE))

    file_uploaded_id = agent.upload_file(open(output_file, "rb"))

    agent_response = agent.invoke(content=system_prompt, attachments=[file_uploaded_id])

    if output_file != INCOMING_FILE:
        os.remove(output_file)

    while (True):
        print_agent_response(agent_response)
        agent_response = agent.invoke(get_user_prompt())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nдI`ll be back!")
