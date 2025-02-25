const std = @import("std");

const Riff = struct { chunk_id: [4]u8, chunksize: u32, format: [4]u8 };

const FmtChunk = packed struct { audio_format: u16, num_channels: u16, sample_rate: u32, byte_rate: u32, block_align: u16, bits_per_sample: u16 };

const Nil = packed struct {};

const DataChunk = struct { stream: std.fs.File.Reader, len: usize };

const SubChunk = union(enum) { fmt: FmtChunk, nil: Nil, data: DataChunk };

const Wav = struct { fmt: FmtChunk, stream: std.fs.File.Reader, len: usize };

const Decoder = struct {
    stream: std.fs.File.Reader,

    fn init(stream: std.fs.File.Reader) Decoder {
        return .{ .stream = stream };
    }

    fn decode(self: *const Decoder) !Wav {
        var riff: Riff = undefined;
        var n = try self.stream.readAll(&riff.chunk_id);
        std.debug.assert(n == 4);
        riff.chunksize = try self.stream.readInt(u32, .little);
        n = try self.stream.readAll(&riff.format);
        std.debug.assert(n == 4);
        var fmt: FmtChunk = undefined;
        var data: DataChunk = undefined;
        fmt = switch (try self.nextSubChunk("fmt ")) {
            .fmt => |f| f,
            else => unreachable,
        };
        while (true) {
            const chunk = try self.nextSubChunk(null);
            switch (chunk) {
                .data => |s| {
                    data = s;
                    break;
                },
                else => {},
            }
        }
        return .{ .fmt = fmt, .stream = data.stream, .len = data.len };
    }

    fn nextSubChunk(self: *const Decoder, subchunk_tag: ?[]const u8) !SubChunk {
        var id: [4]u8 = undefined;
        const n = try self.stream.readAll(id[0..]);
        std.debug.assert(n == 4);
        if (subchunk_tag) |st| {
            if (!std.mem.eql(u8, id[0..], st[0..])) {
                return error.UnexpectedSubChunk;
            }
        }
        var un: SubChunk = undefined;
        const size = try self.stream.readInt(u32, .little);
        if (std.mem.eql(u8, id[0..], "fmt ")) {
            std.debug.assert(size == 16);
            const fmt = try self.stream.readStructEndian(FmtChunk, .little);
            std.debug.assert(fmt.audio_format == 1);
            un = .{ .fmt = fmt };
        } else if (std.mem.eql(u8, id[0..], "data")) {
            un = .{ .data = .{ .stream = self.stream, .len = @intCast(size) } };
        } else {
            try self.stream.skipBytes(size, .{});
            un = .{ .nil = .{} };
        }
        return un;
    }
};

const fftw = @cImport(@cInclude("fftw3.h"));

const ray = @cImport(@cInclude("raylib.h"));

const FFTWComplex = [2]f64;

fn fftwMalloc(T: type, n: usize) []T {
    if (fftw.fftw_malloc(n * @sizeOf(T))) |mem| {
        return @as([*]T, @ptrCast(@alignCast(mem)))[0..n];
    }
    @panic("fft malloc failed");
}

fn fftwFree(mem: anytype) void {
    fftw.fftw_free(mem.ptr);
}

const FFTPlan = struct {
    plan: fftw.fftw_plan,
    pub const Direction = enum { forward, backward };
    fn init(in: []f64, out: []FFTWComplex) FFTPlan {
        return .{ .plan = fftw.fftw_plan_dft_r2c_1d(@intCast(in.len), in.ptr, out.ptr, 1 << 6).? };
    }

    fn execute(self: *FFTPlan) void {
        fftw.fftw_execute(self.plan);
    }

    fn deinit(self: *FFTPlan) void {
        fftw.fftw_destroy_plan(self.plan);
    }
};

var gpa = std.heap.GeneralPurposeAllocator(.{}){};

const c = @import("c.zig");

pub fn main1() !void {
    const cwd = std.fs.cwd();

    var args = std.process.args();
    _ = args.skip();

    var file_name: []const u8 = undefined;

    if (args.next()) |arg| {
        file_name = arg;
    } else {
        std.debug.print("need wav file\n", .{});
        std.debug.print("Usage: \n", .{});
        std.debug.print("   ./bin path_to_wav_file\n", .{});
        std.process.exit(0);
    }

    var file = try cwd.openFile(file_name, .{});
    defer file.close();
    const decoder = Decoder.init(file.reader());
    const riff = try decoder.decode();

    var music_data = try gpa.allocator().alloc(u8, riff.len);

    const n = try riff.stream.readAll(music_data[0..]);
    std.debug.assert(n == riff.len);

    const samples: []i16 = @as([*]i16, @ptrCast(@alignCast(music_data.ptr)))[0 .. music_data.len / 2];

    const a = fftwMalloc(f64, samples.len);

    for (samples, 0..) |_, i| {
        if (i % 2 != 0) continue;
        a[i / 2] = (@as(f64, @floatFromInt(samples[i])) + @as(f64, @floatFromInt(samples[i + 1]))) / 2;
    }
    gpa.allocator().free(music_data);

    const b = fftwMalloc(FFTWComplex, a.len / 2 + 1);
    var plan = FFTPlan.init(a, b);
    plan.execute();
    fftwFree(a);
    var mags = std.ArrayList(f64).init(gpa.allocator());
    defer mags.deinit();

    for (b[0..]) |v| {
        //const nyquist:f64 = @as(f64, @floatFromInt(riff.fmt.sample_rate))/2;
        const mag = std.math.sqrt(std.math.pow(f64, v[0], 2) +
            std.math.pow(f64, v[1], 2)) / @as(f64, @floatFromInt(b.len));
        if (mag == 0.0) {
            break;
        }
        const db = if (mag > 0) 20 * std.math.log10(mag) else -100;
        try mags.append(db);
    }

    fftwFree(b);
    plan.deinit();

    var max_mag: f64 = 0;
    var min_mag: f64 = 0;
    for (mags.items) |m| {
        if (m > max_mag) {
            max_mag = m;
        }

        if (m < min_mag) {
            min_mag = m;
        }
    }

    for (mags.items, 0..) |_, i| {
        mags.items[i] /= max_mag;
    }

    const window_w = 800;
    const window_h = 600;
    c.InitWindow(window_w, window_h, "malware");
    c.SetTargetFPS(60);

    var start_pos: usize = 0;
    while (!c.WindowShouldClose()) {
        c.ClearBackground(c.RED);
        c.BeginDrawing();

        const rem = mags.items.len - start_pos;
        if (rem == 0) break;
        for (0..if (rem > 400) 400 else rem) |i| {
            c.DrawRectangleV(.{ .x = @floatFromInt(i * 2), .y = @floatFromInt(window_h / 2) }, .{ .x = 2, .y = @floatCast(mags.items[i + start_pos] * window_h / 2) }, c.BLACK);
            c.DrawRectangleV(.{ .x = @floatFromInt(i * 2 + 2), .y = @floatFromInt(window_h / 2) }, .{ .x = -2, .y = @floatCast(mags.items[i + start_pos] * -(window_h / 2)) }, c.YELLOW);
        }

        //const wave_rem = samples.len - wave_start_pos;
        //if(wave_rem == 0) break;
        //var old_point = c.Vector2{
        //    .x = 0,
        //    .y = window_h/2
        //};
        //for (0..if (wave_rem > 800) 400 else wave_rem/2) |i| {
        //    var point = c.Vector2 {
        //        .x = @floatFromInt(i*2),
        //        .y = (@as(f32, @floatFromInt(samples[i*2 + wave_start_pos])) /
        //        @as(f32, @floatFromInt(max_ampl)))*(window_h)
        //    };
        //    point.y += window_h/2;
        //    c.DrawLineEx(old_point,point,2, c.GREEN);
        //    old_point = point;
        //}
        //wave_start_pos += 1;

        start_pos += 1;
        c.EndDrawing();
    }

    c.CloseWindow();
}

const pulse = @cImport(@cInclude("pulse/simple.h"));
const pulse_err = @cImport(@cInclude("pulse/error.h"));


pub fn main2() !void {
 
    const cwd = std.fs.cwd();

    var args = std.process.args();
    _ = args.skip();

    var file_name: []const u8 = undefined;

    if (args.next()) |arg| {
        file_name = arg;
    } else {
        std.debug.print("need wav file\n", .{});
        std.debug.print("Usage: \n", .{});
        std.debug.print("   ./bin path_to_wav_file\n", .{});
        std.process.exit(0);
    }

    var file = try cwd.openFile(file_name, .{});
    defer file.close();
    const decoder = Decoder.init(file.reader());
    const riff = try decoder.decode();

    var music_data = try gpa.allocator().alloc(u8, riff.len);

    const n = try riff.stream.readAll(music_data[0..]);
    std.debug.assert(n == riff.len);

    const samples: []i16 = @as([*]i16, @ptrCast(@alignCast(music_data.ptr)))[0 .. music_data.len / 2];
    //_ = samples;

    const a = fftwMalloc(f64, samples.len);
    for (samples, 0..) |_, i| {
        if (i % 2 != 0) continue;
        a[i / 2] = (@as(f64, @floatFromInt(samples[i])));
        // + @as(f64, @floatFromInt(samples[i + 1]))) / 2;
    }
    gpa.allocator().free(music_data);

    std.debug.print("samples len: {} {}\n", .{ samples.len, (samples.len - (samples.len % 64)) });

    const window_w = 800;
    const window_h = 600;
    c.InitWindow(window_w, window_h, "malware");
    c.SetTargetFPS(60);

    const outbuf = fftwMalloc(FFTWComplex, 513);
    //var fft_buf = fftwMalloc(T: type, n: usize)
    var mags: [513]f64 = undefined;

    var start_pos: usize = 10000;
    while (!c.WindowShouldClose()) {
        c.ClearBackground(c.BLACK);
        c.BeginDrawing();

        var plan = FFTPlan.init(a[start_pos..][0..64], outbuf);
        plan.execute();

        plan.deinit();

        var max_mag: f64 = 0;

        for (outbuf[0..], 0..) |v, i| {
            //const nyquist:f64 = @as(f64, @floatFromInt(riff.fmt.sample_rate))/2;
            const mag = std.math.sqrt(std.math.pow(f64, v[0], 2)) /
                @as(f64, @floatFromInt(outbuf.len));
            if (mag == 0.0) {
                break;
            }
            const db = if (mag > 0) 20 * std.math.log10(mag) else -100;
            mags[i] = db;
            if (db > max_mag) max_mag = db;
        }

        var sum: f64 = 0;
        for (0..mags.len) |i| {
            mags[i] /= max_mag;
            sum += mags[i];
        }

        const mean = sum / mags.len;

        for (0..16) |i| {
            const height = @as(f32, @floatCast(mean * -(window_h / 8))) * (@as(f32, @floatFromInt(i + 16)) / @as(f32, @floatFromInt(32)));
            const height2 = @as(f32, @floatCast(mean * -(window_h / 8))) * (@as(f32, @floatFromInt((32 - i) - 1)) / @as(f32, @floatFromInt(32)));
            const thick1 = 16 * (@as(f32, @floatFromInt(i + 16)) / @as(f32, @floatFromInt(32)));
            const thick2 = 16 * (@as(f32, @floatFromInt((32 - i) - 1)) / @as(f32, @floatFromInt(32)));
            const gap1: f32 = (20 * (@as(f32, @floatFromInt(i + 16)) / @as(f32, @floatFromInt(32))));
            const gap2: f32 = (20 * (@as(f32, @floatFromInt((32 - i) - 1)) / @as(f32, @floatFromInt(32)))) + 9;

            std.debug.print("gap1: {}, gap2: {}\n", .{ gap1, gap2 });
            c.DrawRectangleV(.{
                .x = (@as(f32, @floatFromInt(i)) * gap1) + (60 + (mags.len * 10)),
                .y = @floatFromInt(window_h / 2),
            }, .{ .x = -thick1, .y = height }, c.BLUE);
            c.DrawRectangleV(.{ .x = (@as(f32, @floatFromInt(i)) * gap2) + 80, .y = @floatFromInt(window_h / 2) }, .{ .x = -thick2, .y = height2 }, c.BLUE);
        }
        start_pos += 1;
        c.EndDrawing();
        //break;
    }

    c.CloseWindow();
}

const perr = @import("pe.zig");
const pls = @import("p.zig");

pub fn main() !void {
    const cwd = std.fs.cwd();

    var args = std.process.args();
    _ = args.skip();

    var file_name: []const u8 = undefined;

    if (args.next()) |arg| {
        file_name = arg;
    } else {
        std.debug.print("need wav file\n", .{});
        std.debug.print("Usage: \n", .{});
        std.debug.print("   ./bin path_to_wav_file\n", .{});
        std.process.exit(0);
    }

    var file = try cwd.openFile(file_name, .{});
    defer file.close();
    const decoder = Decoder.init(file.reader());
    const riff = try decoder.decode();

    std.debug.print("{}\n", .{riff});

    const samples_begin = try file.getPos();

    const channel_len = riff.len / riff.fmt.num_channels;
    const channel_samples = channel_len / (riff.fmt.bits_per_sample / 8);
    const channel_time = (channel_samples / riff.fmt.sample_rate) * 100;
    std.debug.print("channel len: {}\n", .{channel_len});
    std.debug.print("channel samples: {}\n", .{channel_samples});
    std.debug.print("channel time: {}\n", .{channel_time});

    //const fftbufsize = @divExact(riff.len, 4);

    //std.debug.print("fftbuf size: {}\n", .{fftbufsize});

    //const a = fftwMalloc(f64, fftbufsize);

    //for (0..fftbufsize) |i| {
    //    const sample = try riff.stream.readInt(i16, .little);
    //    a[i] = @floatFromInt(sample);
    //    _ = try riff.stream.readInt(i16, .little);
    //}

    //std.posix.exit(0);

    //for(0..channel_time) |i| {
    //    std.Thread.sleep(std.time.ns_per_ms*100);
    //    const buf_pos:usize = @intFromFloat((@as(f64, @floatFromInt(i))/@as(f64, @floatFromInt(channel_time))) * @as(f64, @floatFromInt(riff.len)));
    //    std.debug.print("time: {}, pos: {}\n", .{i, buf_pos});
    //}

    const outbuf = fftwMalloc(FFTWComplex, 513);
    const inbuff = fftwMalloc(f64, 1024);
    //var fft_buf = fftwMalloc(T: type, n: usize)
    var mags: [513]f64 = undefined;

    const window_w = 800;
    const window_h = 600;
    c.InitWindow(window_w, window_h, "malware");
    c.SetTargetFPS(60);
    const start_time = c.GetTime();
    //var prev_render = start_time;
    var fmt:[128]u8 = undefined;
    const total_time =  @as(f32, @floatFromInt(channel_time));
    //var cp = std.process.Child.init(&.{"audacious", file_name}, gpa.allocator());
    //try cp.spawn();
    //std.Thread.sleep(std.time.ns_per_s * 6);
    while (!c.WindowShouldClose()) {
        c.ClearBackground(c.BLACK);
        c.BeginDrawing();

        const curr_render = c.GetTime();

        if (true) {
            @memset(inbuff, 0);
            const music_time = ((curr_render - start_time) * 100);
            const buf_pos = (music_time / total_time) * @as(f32, @floatFromInt(riff.len));

            @memset(&fmt, 0);

            const time_fmt = try std.fmt.bufPrint(fmt[0..], "{} of {}", .{
                @as(usize, @intFromFloat(music_time/100)), @as(usize, @intFromFloat(total_time/100))});

            c.DrawText(time_fmt.ptr, 500, 200, 40, c.BLUE);

            var pos: usize = @intFromFloat(buf_pos);
            pos &= ~@as(usize, 0b11);
            std.debug.assert(pos % 4 == 0);
            if (pos >= riff.len - 100) break;

            try file.seekTo(samples_begin + pos);
            const rem = riff.len - pos;


            std.debug.assert(rem % 4 == 0);

            const rem_4 = @divExact(rem, 4);

            for(0..if(rem_4 >= 1024) 1024 else rem_4) |i| {
                inbuff[i] = @floatFromInt(try file.reader().readInt(i16, .little));
                _ = try file.reader().readInt(i16, .little);
            }
            //std.debug.print("pos: {}\n", .{pos});
            var plan = FFTPlan.init(inbuff, outbuf);
            plan.execute();
            plan.deinit();
            var max_mag: f64 = 0;
            for (outbuf[0..], 0..) |v, i| {
                //const nyquist:f64 = @as(f64, @floatFromInt(riff.fmt.sample_rate))/2;
                const mag = std.math.sqrt(std.math.pow(f64, v[0], 2)) /
                    @as(f64, @floatFromInt(outbuf.len));
                if (mag == 0.0) {
                    break;
                }
                //const db = if (mag > 0) 20 * std.math.log10(mag) else -100;
                mags[i] = mag;
                if (mag > max_mag) max_mag = mag;
            }
            var sum: f64 = 0;
            for (0..1) |i| {
                //mags[i] /= max_mag;
                sum += mags[i];
            }
            var mean = (sum / 1)/max_mag;
            if(mean < std.math.floatMin(f64)) {
                mean = 0;
            }
            //const mean = (mags[0])/max_mag;
            for (0..16) |i| {
                const height = @as(f32, @floatCast(mean * (window_h / 10))) * (@as(f32, @floatFromInt(i + 16)) / @as(f32, @floatFromInt(32)));
                const height2 = @as(f32, @floatCast(mean * (window_h / 10))) * (@as(f32, @floatFromInt((32 - i) - 1)) / @as(f32, @floatFromInt(32)));
                const thick1 = 18 * (@as(f32, @floatFromInt(i + 16)) / @as(f32, @floatFromInt(32)));
                const thick2 = 18 * (@as(f32, @floatFromInt((32 - i) - 1)) / @as(f32, @floatFromInt(32)));
                const gap1: f32 = (20 * (@as(f32, @floatFromInt(i + 16)) / @as(f32, @floatFromInt(32))));
                const gap2: f32 = (20 * (@as(f32, @floatFromInt((32 - i) - 1)) / @as(f32, @floatFromInt(32)))) + 9;
                //std.debug.print("gap1: {}, gap2: {}\n", .{ gap1, gap2 });

                //if(height < 0)

                
                const bars:usize = @intFromFloat(height/5);
                const bars2:usize = @intFromFloat(height2/5);


                for(0..bars, 0..) |b, s| {
                c.DrawRectangleV(.{
                    .x = (@as(f32, @floatFromInt(i)) * gap1) + (60 + (320))+@as(f32, @floatFromInt(s)),
                    .y = @floatFromInt((window_h / 2)-(b*5)),
                }, .{ .x = thick1, .y = 4 }, c.BLUE);
                } 

                for(0..bars2, 0..) |b, s| {
                c.DrawRectangleV(.{ 
                    .x = (@as(f32, @floatFromInt(i)) * gap2) + 80-@as(f32, @floatFromInt(s)),
                    .y = @floatFromInt((window_h / 2)-(b*5)) },
                     .{ .x = thick2, .y = 4 }, c.BLUE);
                }
            }

            //std.debug.print("frame time: {}\n", .{pos});
        }
        c.EndDrawing();
    }

    c.CloseWindow();
}

pub fn maiun() !void {
    const window_w = 800;
    const window_h = 600;
    c.InitWindow(window_w, window_h, "malware");
    c.SetTargetFPS(60);
    while (!c.WindowShouldClose()) {
        c.ClearBackground(c.BLACK);
        c.BeginDrawing();

        std.debug.print("frame time: {}\n", .{c.GetTime() * 100});

        c.EndDrawing();
    }

    c.CloseWindow();
}
