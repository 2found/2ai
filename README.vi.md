# 2ai

**Nền tảng AI cho hệ thống cần model, công cụ và agent.**

[English](README.md) · [Tiếng Việt](README.vi.md) · [2found](https://2found.dev)

2ai là thư viện Go để xây ứng dụng AI trên cùng hợp đồng provider, vòng lặp
agent native và telemetry. Đây là nền tảng dùng bởi
[2agent / Soot](https://github.com/2found/2agent) và AgentRay. Ứng dụng sử dụng
thư viện sở hữu cấu hình, credential, quyền truy cập và dữ liệu bền vững.

## Bắt đầu trong một phút

Cần **Go 1.25 trở lên**. Để truy cập repo private, xác thực GitHub và đặt
`GOPRIVATE=github.com/2found/*` trước khi tải module.

```sh
git clone https://github.com/2found/2ai.git
cd 2ai
go mod download
go run ./examples/scripted
```

Kết quả: `Hello from 2ai.` Ví dụ dùng agent native với provider giả lập;
không cần API key, dịch vụ bên ngoài hay JavaScript runtime.
Đọc [ví dụ đầy đủ](examples/scripted/main.go), rồi xem API công khai:

```sh
go doc ./agentcore.Config
go doc ./ai.FallbackProvider
go doc ./telemetry.NewInMemory
```

Thêm 2ai vào Go module của bạn:

```sh
go get github.com/2found/2ai@main
```

Go ghi phiên bản theo commit vào `go.mod`. Commit cả `go.mod` và `go.sum`;
pin phiên bản đó khi triển khai. Import các package như
`github.com/2found/2ai/ai` và `github.com/2found/2ai/agentcore`.

## Chọn đúng lớp cần dùng

| Package | Dùng cho | Tài liệu |
| --- | --- | --- |
| `ai`, `ai/protocol` | Transcript, stream provider, OAuth và retry/fallback | [AI](ai/README.md) |
| `agentcore` | Ghép agent, giới hạn, policy và quyền gọi tool | [AgentCore](agentcore/README.md) |
| `agentcore/engine` | Vòng lặp native và event stream cấp thấp | [Engine](agentcore/engine/README.md) |
| `agentcore/host` | Checkpoint và tiện ích chung cho host | `go doc ./agentcore/host` |
| `agentcore/plugins` | Goal, memory, todo, job, subagent và guard | [Plugins](agentcore/plugins/README.md) |
| `telemetry`, `telemetry/export`, `telemetry/llm` | Ghi span và trace model đã kết thúc | [Telemetry](telemetry/README.md) |
| `credential` | Vault trong bộ nhớ và tham chiếu secret | `go doc ./credential` |
| `sandbox` | File, shell, HTTP và công cụ thực thi do host cấp | `go doc ./sandbox` |
| `jsonjs` | JSON không mất dữ liệu và identity của value | `go doc ./jsonjs` |
| `testing/agentcore` | Fixture native session cho test của ứng dụng | `go doc ./testing/agentcore` |

Nền tảng này không chứa application server, database sản phẩm hay CLI.
Dùng [2agent](https://github.com/2found/2agent) để có workspace, hội thoại,
lịch chạy, pack và app của Soot.

## Những hợp đồng cần giữ

- Provider giữ thứ tự nội dung, thinking signature, tool ID và usage.
- AI sở hữu retry/fallback có giới hạn. Khi đã có output hiển thị, cancellation
  hoặc lỗi host, fallback dừng; host gắn callback lưu lựa chọn và usage.
- AgentCore sở hữu một vòng lặp model. Engine nằm dưới policy ứng dụng;
  plugin mở rộng composition, không tạo runtime thứ hai.
- Host giải credential và cấp quyền tool. Cài capability không tự cấp quyền.
  Thư mục workspace không tạo môi trường cách ly hệ điều hành.
- Checkpoint native và identity của value có quy tắc sở hữu. Đọc tài liệu
  package trước khi sao chép state hoặc xử lý event đồng thời.

Các transport native được cấu hình gồm Codex, Claude Code, Antigravity, Devin
và endpoint tương thích OpenAI, bao gồm Gemini. Mức hỗ trợ tùy provider;
[tài liệu AI](ai/README.md) là hợp đồng, không phải cam kết hỗ trợ mọi model
trong catalog.

## Phát triển và kiểm chứng

```sh
go test -race ./...
go vet ./...
go run ./examples/scripted
```

Kiểm tra thông thường dùng fixture ghi sẵn, provider giả lập và HTTP server local.
Test gọi provider thật hoặc Docker được bỏ qua khi chưa cung cấp điều kiện chạy
rõ ràng. Suite kiểm tra lỗi, cancellation, quyền, usage khi retry và concurrency.
Không cần checkout repo bên cạnh hay `replace` local. Khi sửa nhiều repo cùng
lúc, dùng Go workspace không commit.

## Dành cho coding agent

Đọc [AGENTS.md](AGENTS.md) trước để biết ranh giới sở hữu, lệnh kiểm tra và
quy tắc credential. Tiếp theo đọc README gần package và boundary test; dùng
`go doc` để xem type công khai. Giữ runtime behavior ở đúng lớp sở hữu;
kiến thức workload thuộc config, skill hoặc tool của ứng dụng sử dụng thư viện.
Cập nhật đồng thời README này và [bản tiếng Anh](README.md) khi đổi cách cài
đặt hoặc hành vi hỗ trợ.

## Nguồn gốc và giấy phép

Tách từ các package AI dùng chung của AgentRay. [SOURCE.json](SOURCE.json) ghi
commit nguồn, hash file gốc và ánh xạ đường dẫn. `credential`, `sandbox`,
JSON primitive và test helper đi cùng ba package chính để ứng dụng build độc lập.
Application và analytics của AgentRay ở lại repo riêng.

[MIT](LICENSE), giữ nguyên các notice gốc. Thành phần kế thừa Pi, TypeBox,
partial-json và Unicode giữ notice riêng cạnh source;
[nguồn gốc Pi](third_party/pi/README.md) ghi hash source upstream.

Một project của [2found](https://2found.dev). **From idea to company.**
