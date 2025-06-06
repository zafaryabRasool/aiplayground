"""add rag technique to chat

Revision ID: 40c156115b24
Revises: d58cdb4a30f8
Create Date: 2024-09-25 23:32:08.198637

"""
# pylint: disable=no-member, invalid-name
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "40c156115b24"
down_revision: Union[str, None] = "d58cdb4a30f8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Migration to add RAG technique to chat
    """
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column(
        "chat",
        sa.Column(
            "rag_technique",
            sa.Enum("NONE", "VECTOR", "GRAPH", name="ragtechnique"),
            nullable=False,
        ),
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    """
    Migration to remove RAG technique from chat
    """
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column("chat", "rag_technique")
    # ### end Alembic commands ###
