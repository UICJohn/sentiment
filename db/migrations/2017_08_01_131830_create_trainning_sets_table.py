from orator.migrations import Migration


class CreateTrainningSetsTable(Migration):

  def up(self):
    with self.schema.create('trainning_sets') as table:
      table.increments('id')
      table.text('words')
      table.boolean("positive")
      table.text('word_ids').nullable()
      table.index('positive')
      table.index('word_ids')
      table.timestamps()

  def down(self):
    """
    Revert the migrations.
    """
    pass
