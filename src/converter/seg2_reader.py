#! /usr/bin/python3
"""
SEG2 file format reader

This module provides the :class:`SEG2Reader` class for reading SEG-2
format seismic data files. The implementation was extracted verbatim from
:mod:`src.converter.seg2` as part of the SEG-2 reading-layer separation
slice. The runtime behavior of :class:`SEG2Reader` is unchanged from the
original implementation.

Backward compatibility is preserved by re-exporting :class:`SEG2Reader`
from :mod:`src.converter.seg2`, so existing imports such as
``from src.converter.seg2 import SEG2Reader`` continue to work.
"""

import copy
import os
import struct

import numpy as np


class SEG2Reader:
    def __uint16t_read(self, ba, p):
        return int.from_bytes(ba[p : p + 2], byteorder="little", signed=False)

    def __int32t_read(self, ba, p):
        return int.from_bytes(ba[p : p + 4], byteorder="little", signed=True)

    def __uint32t_read(self, ba, p):
        return int.from_bytes(ba[p : p + 4], byteorder="little", signed=False)

    def __uint16t_str(self, val):
        return str("0x%04x" % val)

    def __uint32t_str(self, val):
        return str("0x%08x" % val)

    ######################################################################
    ## SEG2ファイルを読み込んでチャンネルごとのデータ位置を特定する	 	##
    ######################################################################
    def __seg2_get_pointer_from_file_desc(self):
        p = 0
        # Find Descriptor
        if self.__uint16t_read(self.__data, p) != 0x3A55:
            print("File have no File Descriptor!")
            return []
        ##print("File descriptor block")
        ##print("\tStart from " + uint32t_str(p))
        p = p + 0x02
        # revision
        revision = self.__uint16t_read(self.__data, p)
        ##print("\tRevision: " + uint16t_str(revision))
        p = p + 0x02
        # M
        m = self.__uint16t_read(self.__data, p)
        ##print("\tM=Size of Trace Pointer Sub-block(in bytes): " + uint16t_str(m))
        p = p + 0x02
        # N
        n = self.__uint16t_read(self.__data, p)
        ##print("\tN=Number of traces in this file: " + uint16t_str(n))
        p = p + 0x02
        ## Ignore 0x08-0x0F
        p = p + 0x08
        ## Ignore 0x10-0x1F
        p = p + 0x10
        # List of Pointer
        list_pointer = []
        for trace in range(0, n):
            ## Pointerの場所がデータ位置
            pointer = self.__uint32t_read(self.__data, p)
            list_pointer.append(pointer)
            ##print("\t\tPointer[" + uint16t_str(trace) + "]:" + uint32t_str(pointer))
            p = p + 0x04
        # Free Format String
        return list_pointer

    ######################################################################
    ## 指定アドレスから始まるチャンネルデータを波形データ(float)に変換	##
    ######################################################################
    def __seg2_get_wav_from_trace_descriptor(self, pointer, log=False):
        ret = {}
        p = pointer
        # Find Descriptor
        if self.__uint16t_read(self.__data, p) != 0x4422:
            return []
        p = p + 0x02
        # X
        x = self.__uint16t_read(self.__data, p)
        p = p + 0x02
        ret["x"] = x
        # Y
        y = self.__uint32t_read(self.__data, p)
        p = p + 0x04
        ret["y"] = y
        # NS
        ns = self.__uint32t_read(self.__data, p)
        p = p + 0x04
        ret["ns"] = ns
        # DataFormat
        df = self.__data[p]
        p = p + 0x04
        # Reserved
        p = p + 0x10
        # Parse Strings
        ret["free"] = self.__seg2_parse_string(p, pointer + x)
        ## Skip to Data
        p = pointer + x
        # Read Data
        wav = []
        for sample in range(0, ns):
            if df == 0x01:
                wav.append(struct.unpack("<h", self.__data[p : p + 2])[0])
                p = p + 0x02
            elif df == 0x02:
                wav.append(struct.unpack("<i", self.__data[p : p + 4])[0])
                p = p + 0x04
            elif df == 0x03:
                print("\t\t\t[%d]Unsupported format." % d)
                break
            elif df == 0x04:
                wav.append(struct.unpack("<f", self.__data[p : p + 4])[0])
                p = p + 0x04
            elif df == 0x05:
                wav.append(struct.unpack("<d", self.__data[p : p + 8])[0])
                p = p + 0x08
        ret["wav"] = wav
        # logging
        if log:
            print("\tX=Size of this block: " + self.__uint16t_str(x))
            print("Trace descriptor block")
            print("\tStart from " + self.__uint32t_str(p))
            print("\tY=Sizes of Corresponding Data Block: " + self.__uint32t_str(y))
            print("\tNS=Number of samples in Data Block: " + self.__uint32t_str(ns))
            if df == 0x01:
                print("\tDataFormat: 16-bit fixed point")
            elif df == 0x02:
                print("\tDataFormat: 32-bit fixed point")
            elif df == 0x03:
                print("\tDataFormat: 20-bit floating point(SEG-D)")
            elif df == 0x04:
                print("\tDataFormat: 32-bit floating point(IEEE)")
            elif df == 0x05:
                print("\tDataFormat: 64-bit floating point(IEEE)")
        return ret

    def __seg2_get_factor_to_specific_pointer(self, pointer_to, pointer_from=0):
        # 0番目からpointer_toまでの文字列を取得
        ret = {}
        p = pointer_from
        # Parse Strings
        ret["free"] = self.__seg2_parse_string(p, pointer_to)
        return ret

    ######################################################################
    ## 指定アドレスから始まるチャンネルデータを反転させる				##
    ######################################################################
    def __seg2_inversion_bytes_data(self, inversion, pointer):
        p = pointer
        # Find Descriptor
        if self.__uint16t_read(self.__data, p) != 0x4422:
            return []
        p = p + 0x02
        # X
        x = self.__uint16t_read(self.__data, p)
        p = p + 0x02
        # Y
        y = self.__uint32t_read(self.__data, p)
        p = p + 0x04
        # NS
        ns = self.__uint32t_read(self.__data, p)
        p = p + 0x04
        # DataFormat
        df = self.__data[p]
        p = p + 0x04
        # Reserved
        p = p + 0x10
        # Parse Strings
        ## skip
        ## Skip to Data
        p = pointer + x
        # Read Data
        wav = 0.0
        for sample in range(0, ns):
            if df == 0x01:
                print("\t\t\t[%d]Unsupported format." % d)
                break
            elif df == 0x02:
                wav = struct.unpack("<i", self.__data[p : p + 4])[0]
                wav = wav * (-1)
                bs = struct.pack("<i", wav)
                inversion[p + 0] = bs[0]
                inversion[p + 1] = bs[1]
                inversion[p + 2] = bs[2]
                inversion[p + 3] = bs[3]
                p += 0x04
            elif df == 0x03:
                print("\t\t\t[%d]Unsupported format." % d)
                break
            elif df == 0x04:
                print("\t\t\t[%d]Unsupported format." % d)
                break
            elif df == 0x05:
                print("\t\t\t[%d]Unsupported format." % d)
                break
        return inversion

    ######################################################################
    ## 指定アドレスから始まるチャンネルデータを指定データにスワップする	##
    ######################################################################
    def __seg2_swap_data(self, origin, swap_wav, pointer):
        p = pointer
        # Find Descriptor
        if self.__uint16t_read(self.__data, p) != 0x4422:
            return []
        p = p + 0x02
        # X
        x = self.__uint16t_read(self.__data, p)
        p = p + 0x02
        # Y
        y = self.__uint32t_read(self.__data, p)
        p = p + 0x04
        # NS
        ns = self.__uint32t_read(self.__data, p)
        p = p + 0x04
        # DataFormat
        df = self.__data[p]
        p = p + 0x04
        # Reserved
        p = p + 0x10
        # Parse Strings
        ## skip
        ## Skip to Data
        p = pointer + x
        for sample in range(0, ns):
            if df == 0x01:
                print("\t\t\t[%d]Unsupported format." % d)
                break
            elif df == 0x02:
                bs = struct.pack("<i", swap_wav[sample])
                origin[p + 0] = bs[0]
                origin[p + 1] = bs[1]
                origin[p + 2] = bs[2]
                origin[p + 3] = bs[3]
                p += 0x04
            elif df == 0x03:
                print("\t\t\t[%d]Unsupported format." % d)
                break
            elif df == 0x04:
                bs = struct.pack("<f", swap_wav[sample])
                origin[p + 0] = bs[0]
                origin[p + 1] = bs[1]
                origin[p + 2] = bs[2]
                origin[p + 3] = bs[3]
                p += 0x04
                break
            elif df == 0x05:
                print("\t\t\t[%d]Unsupported format." % d)
                break
        return origin

    ######################################################################
    ## WAV LPCM Integer 24bit形式で保存									##
    ######################################################################
    def __make_int24lpcm_wav(self, wav):
        ret = bytes()
        # RIFF
        ret += b"RIFF"
        # Lenght
        ret += struct.pack("<I", len(wav) * 3 - 8)
        # WAVE
        ret += b"WAVE"
        # fmt_
        ret += b"fmt "
        # LPCM
        ret += struct.pack("<I", 0x10)
        # format:LPCM
        ret += struct.pack("<H", 0x0001)
        # channel:1ch
        ret += struct.pack("<H", 0x0001)
        # SamplingRatec1000Hz
        ret += struct.pack("<I", self.__frequency)
        # bps
        ret += struct.pack("<I", self.__frequency * 3)
        # block size
        ret += struct.pack("<H", 0x03)
        # bit depth
        ret += struct.pack("<H", 24)
        # Data
        ret += b"data"
        ret += struct.pack("<I", len(wav) * 3)
        for w in wav:
            # div by 24bit integer max
            temp = struct.pack("<i", int(w))
            ret += temp[0:1]
            ret += temp[1:2]
            ret += temp[2:3]
        return

    ######################################################################
    ## WAV LPCM Integer 32bit形式で保存								     ##
    ######################################################################
    def __make_int32lpcm_wav(self, wav):
        ret = bytes()
        # RIFF
        ret += b"RIFF"
        # Lenght
        ret += struct.pack("<I", len(wav) * 4 - 8)
        # WAVE
        ret += b"WAVE"
        # fmt_
        ret += b"fmt "
        # LPCM
        ret += struct.pack("<I", 0x10)
        # format:LPCM
        ret += struct.pack("<H", 0x0001)
        # channel:1ch
        ret += struct.pack("<H", 0x0001)
        # SamplingRatec1000Hz
        ret += struct.pack("<I", self.__frequency)
        # bps
        ret += struct.pack("<I", self.__frequency * 4)
        # block size
        ret += struct.pack("<H", 0x04)
        # bit depth
        ret += struct.pack("<H", 32)
        # Data
        ret += b"data"
        ret += struct.pack("<I", len(wav) * 4)
        for w in wav:
            # mul in order to up from 24bit to 32bit
            ret += struct.pack("<i", int(w * 0x100))
        return

    ######################################################################
    ## WAV IEEE Float 32bit形式で保存									##
    ######################################################################
    def __make_float32ieee_wav(self, wav):
        ret = bytes()
        # RIFF
        ret += b"RIFF"
        # Lenght
        ret += struct.pack("<I", 56 + len(wav) * 4 - 8)
        # WAVE
        ret += b"WAVE"
        # fmt_
        ret += b"fmt "
        # LPCM
        ret += struct.pack("<I", 0x10)
        # format:IEEE float
        ret += struct.pack("<H", 0x0003)
        # channel:1ch
        ret += struct.pack("<H", 0x0001)
        # SamplingRatec1000Hz
        ret += struct.pack("<I", self.__frequency)
        # bps
        ret += struct.pack("<I", self.__frequency * 4)
        # block size
        ret += struct.pack("<H", 0x04)
        # bit depth
        ret += struct.pack("<H", 32)
        # FACT(extra header)
        ret += b"fact"
        ret += struct.pack("<I", 0x04)
        ret += struct.pack("<I", 0x03E8)
        # Data
        ret += b"data"
        ret += struct.pack("<I", len(wav) * 4)
        for w in wav:
            # div by 24bit integer max
            # unit = 0x800000
            # ret += struct.pack("<f", w/unit)
            ret += struct.pack("<f", w)
        return ret

    ###################################################################
    ## SEG2のフリーストリングから文字列型のDictを取得する		         ##
    ###################################################################
    def __seg2_parse_string(self, p_from, p_to):
        list = []
        for bytes in self.__data[p_from:p_to].split(b"\0"):
            try:
                st = bytes.decode()
                if len(st) > 1:
                    list.append(st)
            except:
                pass
        return list

    ## 周波数情報を取得する
    def get_frequency(self):
        return self.__frequency

    ## チャンネル数を取得する
    def get_max_ch(self):
        return self.__maxch

    ## 指定したCHのWAVデータを取得する
    def get_wav_data(self, ch):
        if ch < self.__maxch:
            dict = self.__seg2_get_wav_from_trace_descriptor(self.__pointer_list[ch])
            return dict["wav"]
        return None

    ## 全チャンネルをNumpyArray型で出力する
    def get_all_numpy_array(self):
        ret = []
        for ch in range(self.__maxch):
            dict = self.__seg2_get_wav_from_trace_descriptor(self.__pointer_list[ch])
            ret.append(dict["wav"])
        ret = np.array(ret).transpose()
        return ret

    ## WAVファイルを出力する
    def output_wav(self, filename, dirname):
        ret = []
        # 各チャンネルごとのデータをWAVに変換
        for ch in range(0, len(self.__pointer_list)):
            # float型の配列(float[])で波形データを取得
            dict = self.__seg2_get_wav_from_trace_descriptor(self.__pointer_list[ch])
            # フォルダが無ければ作成
            os.makedirs(filename, exist_ok=True)
            # float32形式で出力
            bytes_data = self.__make_float32ieee_wav(dict["wav"])
            outpath = self.__output_file(
                filename + "_" + str("%02d" % ch), ".wav", dirname, bytes_data
            )
            ret.append(outpath)
        return ret

    # ファイル出力用内部関数
    def __output_file(self, filename, ext, dirname, bytes_data):
        if filename is None:
            return ""
        export_name = ""
        if dirname is not None:
            os.makedirs(dirname, exist_ok=True)
            export_name += dirname + "/"
        export_name += filename + ext
        with open(export_name, "wb") as f:
            f.write(bytes_data)
            f.close()
        return export_name

    def output_seg2_without_selected_chs(self, filename, dirname, chs):
        ret = bytes()
        p = 0
        # Copy BlockID + Revision Number
        ret += self.__data[p : p + 4]
        p += 4
        # Copy M
        ret += self.__data[p : p + 2]
        p += 2
        # overrided N
        new_ch_num = self.__maxch - len(chs)
        ret += struct.pack("<H", new_ch_num)
        p += 2
        # Copy until 32 bytes
        ret += self.__data[p : p + 24]
        p += 24
        # data will start from ???
        start_from = self.__pointer_list[0] - 4 * len(chs)
        new_pointer_list = []
        for ch in range(self.__maxch):
            if ch not in chs:
                new_pointer_list.append(start_from)
                if (ch + 1) < self.__maxch:
                    start_from += self.__pointer_list[ch + 1] - self.__pointer_list[ch]
                else:
                    start_from += len(self.__data) - self.__pointer_list[ch]
        # Copy Trace Pointer Subblock
        for pointer in new_pointer_list:
            ret += struct.pack("<I", pointer)
        p = 32 + (4 * self.__maxch)
        # Copy String Area
        ret += self.__data[p : self.__pointer_list[0]]
        # Copy Trace Data
        for ch in range(self.__maxch):
            if ch not in chs:
                if (ch + 1) < self.__maxch:
                    ret += self.__data[
                        self.__pointer_list[ch] : self.__pointer_list[ch + 1]
                    ]
                else:
                    ret += self.__data[self.__pointer_list[ch] :]

        # ファイルの出力
        if filename is not None:
            return self.__output_file(filename, ".sg2", dirname, ret)
        else:
            return ret

    def output_seg2_with_selected_chs(self, filename, dirname, chs):
        ret = bytes()
        p = 0
        # Copy BlockID + Revision Number
        ret += self.__data[p : p + 4]
        p += 4
        # Copy M
        ret += self.__data[p : p + 2]
        p += 2
        # overrided N
        new_ch_num = len(chs)
        ret += struct.pack("<H", new_ch_num)
        p += 2
        # Copy until 32 bytes
        ret += self.__data[p : p + 24]
        p += 24
        # data will start from ???
        start_from = self.__pointer_list[0] - 4 * (self.__maxch - len(chs))
        new_pointer_list = []
        for ch in range(self.__maxch):
            if ch in chs:
                new_pointer_list.append(start_from)
                if (ch + 1) < self.__maxch:
                    start_from += self.__pointer_list[ch + 1] - self.__pointer_list[ch]
                else:
                    start_from += len(self.__data) - self.__pointer_list[ch]
        # Copy Trace Pointer Subblock
        for pointer in new_pointer_list:
            ret += struct.pack("<I", pointer)
        p = 32 + (4 * self.__maxch)
        # Copy String Area
        ret += self.__data[p : self.__pointer_list[0]]
        # Copy Trace Data
        for ch in range(self.__maxch):
            if ch in chs:
                if (ch + 1) < self.__maxch:
                    ret += self.__data[
                        self.__pointer_list[ch] : self.__pointer_list[ch + 1]
                    ]
                else:
                    ret += self.__data[self.__pointer_list[ch] :]
        # ファイルの出力
        if filename is not None:
            return self.__output_file(filename, ".sg2", dirname, ret)
        else:
            return ret

    # XYZのSEG2ファイルを出力する
    def ouput_xyz_seg2(self, filename, dirname):
        max = self.__maxch - self.__maxch % 12  # 12で割ったあまり
        self.output_seg2_with_selected_chs(filename + "_x", dirname, range(0, max, 3))
        self.output_seg2_with_selected_chs(filename + "_y", dirname, range(1, max, 3))
        self.output_seg2_with_selected_chs(filename + "_z", dirname, range(2, max, 3))

    # 48chのSEG"ファイルを分割する
    def output_2D_seg2(self, filename, dirname):
        # トリガーレス
        max = self.__maxch - self.__maxch % 24
        self.output_seg2_with_selected_chs(filename + "_1-24", dirname, range(0, 24))
        self.output_seg2_with_selected_chs(filename + "_24-48", dirname, range(24, 48))

    # XYZのSEG2Readerクラスを出力する
    def get_xyz_seg2reader(self):
        max = self.__maxch - self.__maxch % 24
        x = SEG2Reader(self.output_seg2_with_selected_chs(None, None, range(0, max, 3)))
        y = SEG2Reader(self.output_seg2_with_selected_chs(None, None, range(1, max, 3)))
        z = SEG2Reader(self.output_seg2_with_selected_chs(None, None, range(2, max, 3)))
        return (x, y, z)

    # トリガーを除いたSEG2ファイルを出力する
    def ouput_trigerless_seg2(self, filename, dirname):
        max = self.__maxch - self.__maxch % 24
        self.output_seg2_with_selected_chs(
            filename + "_triggerless", dirname, range(0, max)
        )

    # トリガーを除いたSEG2Readerファイルを出力する
    def get_trigerless_seg2reader(self):
        max = self.__maxch - self.__maxch % 24
        triggerless = SEG2Reader(
            self.output_seg2_with_selected_chs(None, None, range(0, max))
        )
        return triggerless

    ## 波形を反転させたWAVファイルを出力する
    def output_inversion_multi_ch(self, filename, dirname, chs):
        inversion = bytearray(copy.deepcopy(self.__data))
        for ch in chs:
            pointer = self.__pointer_list[ch]
            self.__seg2_inversion_bytes_data(inversion, pointer)

        # ファイルの出力
        ret = self.__output_file(filename + "_inversion", ".sg2", dirname, inversion)
        return ret

    ## CSVファイルの波形データで上書きしたSEG2ファイルを出力する
    ## filename に出力ファイル名
    ## csv_path にcsvファイルを入力する
    def output_seg2_data_overrided_by_csv(self, filename, dirname, csv_path):
        data = np.transpose(np.genfromtxt(csv_path, delimiter=","))
        ## Nanデータがデータが出ないように成形する
        while np.isnan(data).any():
            for index in range(len(data)):
                if np.isnan(data[index]).any():
                    data = np.delete(data, index, axis=0)
                    break
        ## 成型後データから情報を取得
        csv_chs = int(len(data))
        csv_sample_num = int(len(data[0]))
        # 周波数等の情報が一致しているか確認する
        seg2_chs = self.__maxch
        seg2_sample_num = len(self.get_wav_data(0))

        if csv_chs > seg2_chs or csv_sample_num > seg2_sample_num:
            print(
                "Error! This seg2 file is not same(smaller) condition as(than) csv file!"
            )
            print("CSV.ChNum:\t" + str(csv_chs))
            print("CSV.SamplingNum:" + str(csv_sample_num))
            print("SG2.ChNum:\t" + str(seg2_chs))
            print("SG2.SamplingNum:" + str(seg2_sample_num))
            return

        # WAVデータを書き換える
        origin = bytearray(copy.deepcopy(self.__data))
        for ch in range(csv_chs):
            wav = np.array(data[ch], dtype="int32").tolist()
            if csv_sample_num < seg2_sample_num:
                for count in range(seg2_sample_num - csv_sample_num):
                    wav.append(0)
            pointer = self.__pointer_list[ch]
            origin = self.__seg2_swap_data(origin, wav, pointer)

        # ファイルの出力
        if filename is not None:
            return self.__output_file(filename + "_CSV", ".sg2", dirname, origin)
        else:
            return bytes(origin)

    ## ORGファイルの波形データで上書きしたSEG2ファイルを出力する
    ## filename に出力ファイル名
    ## org_path にorgファイルを入力する
    def output_seg2_data_overrided_by_org(self, filename, dirname, org_path):
        data = open(org_path, "rb").read()
        strings = data[:0x1800].decode("ASCII").splitlines()
        org_chs = int(strings[6])
        org_sample_num = int(strings[7])
        org_frequency = int(strings[11])

        # 周波数等の情報が一致しているか確認する
        seg2_chs = self.__maxch - self.__maxch % 24
        seg2_sample_num = len(self.get_wav_data(0))
        seg2_frequency = self.get_frequency()

        if (
            org_chs != seg2_chs
            or org_sample_num != seg2_sample_num
            or org_frequency != seg2_frequency
        ):
            print("Error! This seg2 file is not same condition as org file!")
            print("ORG.ChNum:\t" + str(org_chs))
            print("ORG.SamplingNum:" + str(org_sample_num))
            print("ORG.Frequency:\t" + str(org_frequency))
            print("SG2.ChNum:\t" + str(seg2_chs))
            print("SG2.SamplingNum:" + str(seg2_sample_num))
            print("SG2.Frequency:\t" + str(seg2_frequency))
            return

        # ORGファイルからWAVファイルを掘り出す
        # WAVデータを書き換える
        origin = bytearray(copy.deepcopy(self.__data))
        ch = 0
        for offset in range(0x1800, 0xD1800, 0x8000):
            pointer = self.__pointer_list[ch]
            wav = []
            for i in range(0, 0x8000, 4):
                temp = struct.unpack("<i", data[offset + i : offset + i + 4])[0]
                wav.append(temp)
            ch += 1
            origin = self.__seg2_swap_data(origin, wav, pointer)

        # ファイルの出力
        if filename is not None:
            return self.__output_file(filename + "_ORG", ".sg2", dirname, origin)
        else:
            return bytes(origin)

    def get_org_seg2reader(self, org_path):
        org = SEG2Reader(self.output_seg2_data_overrided_by_org(None, None, org_path))
        return org

    ## SEG2の情報をprintで書き出します
    def info(self):
        seg2_chs = self.__maxch - self.__maxch % 12  #######12ch
        seg2_sample_num = len(self.get_wav_data(0))
        seg2_frequency = self.get_frequency()
        print("ChsWithTRIG:" + str(self.__maxch))
        print("AvailableCh:" + str(seg2_chs))
        print("SamplingNum:" + str(seg2_sample_num))
        print("Frequency  :" + str(seg2_frequency))

    def get_acquisition_time(self):
        return self.__acquisition_time

    def get_acquisition_date(self):
        return self.__acquisition_date

    def get_delay(self):
        return self.__delay

    def get_descaling_factor(self):
        return self.__descaling_factor

    def get_high_cut_filter(self):
        return self.__high_cut_filter

    def get_low_cut_filter(self):

        return self.__low_cut_filter

    def __init__(self, input):
        if type(input) is bytes:
            self.__data = input
        else:
            self.__data = open(input, "rb").read()
        self.__pointer_list = self.__seg2_get_pointer_from_file_desc()
        self.__maxch = len(self.__pointer_list)
        self.__frequency = 0
        self.__acquisition_time = ""
        self.__acquisition_date = ""

        # 周波数情報を取得
        ret = self.__seg2_get_wav_from_trace_descriptor(self.__pointer_list[0])
        for free in ret["free"]:
            if "SAMPLE_INTERVAL" in free:
                temp = free.split(" ")
                index = temp.index("SAMPLE_INTERVAL")
                interval = float(temp[index + 1])
                self.__frequency = int(1.0 / interval)

            if "DELAY" in free:
                temp = free.split(" ")
                index = temp.index("DELAY")
                delay = float(temp[index + 1])
                self.__delay = delay

            if "DESCALING_FACTOR" in free:
                temp = free.split(" ")
                index = temp.index("DESCALING_FACTOR")
                descaling_factor = float(temp[index + 1])
                self.__descaling_factor = descaling_factor

            if "HIGH_CUT_FILTER" in free:
                temp = free.split(" ")
                index = temp.index("HIGH_CUT_FILTER")
                # tempの中に' 'がある場合、その次の要素を排除する
                if "" in temp:
                    temp.remove("")
                high_cut_filter = [float(temp[index + 1]), float(temp[index + 2])]
                self.__high_cut_filter = high_cut_filter

            if "LOW_CUT_FILTER" in free:
                temp = free.split(" ")
                index = temp.index("LOW_CUT_FILTER")
                # tempの中に' 'がある場合、その次の要素を排除する
                if "" in temp:
                    temp.remove("")
                low_cut_filter = [float(temp[index + 1]), float(temp[index + 2])]
                self.__low_cut_filter = low_cut_filter

        ret_2 = self.__seg2_get_factor_to_specific_pointer(self.__pointer_list[0])
        for free in ret_2["free"]:
            if "ACQUISITION_TIME" in free:
                temp = free.split(" ")
                index = temp.index("ACQUISITION_TIME")
                acquisition_time = str(temp[index + 1])
                self.__acquisition_time = acquisition_time

            if "ACQUISITION_DATE" in free:
                temp = free.split(" ")
                index = temp.index("ACQUISITION_DATE")
                acquisition_date = str(temp[index + 1])
                self.__acquisition_date = acquisition_date


__all__ = ["SEG2Reader"]
