"""Change reasoning content from TEXT to MEDIUMTEXT

Revision ID: d58cdb4a30f8
Revises: f372a6852835
Create Date: 2024-09-15 23:29:05.252391

"""
from typing import Sequence, Union

# pylint: disable=no-member
from alembic import op
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision: str = "d58cdb4a30f8"
down_revision: Union[str, None] = "f372a6852835"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Migration to change data type for Reasoning step content
    """
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column(
        "reasoning_step",
        "content",
        existing_type=mysql.TEXT(),
        type_=mysql.MEDIUMTEXT(),
        existing_nullable=False,
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    """
    Migration to change data type for Reasoning step content
    """
    # ### commands auto generated by Alembic - please adjust! ###
    op.alter_column(
        "reasoning_step",
        "content",
        existing_type=mysql.MEDIUMTEXT(),
        type_=mysql.TEXT(),
        existing_nullable=False,
    )
    # ### end Alembic commands ###
