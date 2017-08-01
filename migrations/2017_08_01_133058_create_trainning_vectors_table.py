from orator.migrations import Migration


class CreateTrainningVectorsTable(Migration):

  def up(self):
    with self.schema.create('trainning_vectors') as table:
      table.increments('id')
      table.integer("batch_id").unsigned()
      table.text('vector_ids')
      table.integer("set_id").unsigned()
      table.foreign('set_id').references("id").on("trainning_sets")
      table.foreign('batch_id').references("id").on("batches")
      table.index('batch_id')
      table.index('vector_ids')
      table.index('set_id')
      table.timestamps()


    def down(self):
        """
        Revert the migrations.
        """
        pass
