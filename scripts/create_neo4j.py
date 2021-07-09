from collections import defaultdict
import numpy as np
import codecs
import json
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost", auth=("neo4j", "password"))


def generate_labels(filename):
    print("generating %s" % filename)
    if filename != "Movies":
        label_name = filename.replace("dbo_", "")
    else:
        label_name = "Movies"
    with driver.session(database="neo4j") as session:
        result = session.run("CREATE CONSTRAINT ON (m: %s) ASSERT m.name IS UNIQUE;" % label_name)

def generate_links(filename):
    print(filename)
    prop_name = filename.replace("dbo_", "")
    movies_added = []
    with codecs.open("datasets\\Movielens1M\\graphs\\%s.edgelist" % filename, 'r', encoding='latin-1') as cine:
        for i, line in enumerate(cine):
            add_to_list = False
            line = line.strip('\n')
            line_split = line.split(" ")
            movie = line_split[0]
            prop = line_split[1]
            movies_added.append(movie)
            query = """
                MERGE (p:Movies {name: $movie})
                MERGE (q:%s {name: $prop})
                CREATE (q)-[:%s]->(p)
                """ % (prop_name, prop_name)
            with driver.session(database="neo4j") as session:
                result = session.run(query, {"movie": movie, "prop": prop})


def generate_user():
    movies_added = []
    with codecs.open("datasets\\Movielens1M\\graphs\\feedback.edgelist", 'r', encoding='latin-1') as feedback:
        for i, line in enumerate(feedback):
            add_to_list = False
            line = line.strip('\n')
            line_split = line.split(" ")
            user_id = line_split[0]
            movie = line_split[1].strip()
            query = """
                MATCH (q:Movies {name: $movie})
                MERGE (p:User {name: $user_id})
                CREATE (p)-[:LIKES]->(q)
                """
            with driver.session(database="neo4j") as session:
                result = session.run(query, {"movie": movie, "user_id": user_id})

# generate_labels("Movies")
# generate_labels("dbo_cinematography")
# generate_labels("dbo_director")
# generate_labels("dbo_distributor")
# generate_labels("dbo_starring")
# generate_labels("dbo_editing")
# generate_labels("dbo_writer")
# generate_labels("dbo_musicComposer")
# generate_labels("dbo_producer")


# generate_links("dbo_cinematography")
# generate_links("dbo_director")
# generate_links("dbo_distributor")
# generate_links("dbo_starring")
# generate_links("dbo_editing")
# generate_links("dbo_writer")
# generate_links("dbo_musicComposer")
# generate_links("dbo_producer")

generate_user()
