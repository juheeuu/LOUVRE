from retriever.louvre import pred_nldb

if __name__ == "__main__":
    args = pred_nldb.parse_args()
    _ = pred_nldb.main(args)
