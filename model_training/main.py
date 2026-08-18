import settings


def main():
    print("===== FLOWER WITH ROSPACE =====")
    runner = settings.MainRunner()

    while not runner.is_done():
        try:
            runner.split_data()
            runner.run_server()
            runner.run_clients()
            runner.life_check()
            print("[Info] If u wanna close them, press Ctrl+C, but I wish u don't do that ouob .\n")

        except KeyboardInterrupt:
            print("\n[Info] I have been interrupt oao .")
            break

        except Exception as e:
            print(f"\n[Error] Exception: {e}")
            break

        finally:
            print("[Info] closing all process...")
            runner.shut_down()

        runner.next()


if __name__ == "__main__":
    main()