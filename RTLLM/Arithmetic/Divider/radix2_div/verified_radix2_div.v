`timescale 1ns/1ps
module radix2_div(
    input wire clk,
    input wire rst,
    input wire [7:0] dividend,    
    input wire [7:0] divisor,    
    input wire sign,       

    input wire opn_valid,   
    output reg res_valid,   
    input wire res_ready,   
    output wire [15:0] result
);

    reg [7:0] dividend_save, divisor_save;
    reg [15:0] SR;                  
    reg [8 :0] NEG_DIVISOR;        
    wire [7:0] REMAINER, QUOTIENT;
    assign REMAINER = SR[15:8];
    assign QUOTIENT = SR[7: 0];

    // Keep the original signal name for testbench compatibility
    wire [7:0] divident_abs;
    wire [8:0] divisor_abs;
    assign divident_abs = dividend_abs_reg;
    assign divisor_abs = neg_divisor_reg;
    
    wire [7:0] remainer, quotient;

    // Fixed: Explicit sign correction with proper 2's complement
    assign remainer = (sign & dividend_save[7]) ? ((~REMAINER) + 8'b00000001) : REMAINER;
    assign quotient = (sign & (dividend_save[7] ^ divisor_save[7])) ? ((~QUOTIENT) + 8'b00000001) : QUOTIENT;
    assign result = {remainer, quotient};

    wire CO;
    wire [8:0] sub_result;
    wire [8:0] mux_result;

    assign {CO,sub_result} = {1'b0,REMAINER} + NEG_DIVISOR;
    assign mux_result = CO ? sub_result : {1'b0,REMAINER};

    reg [3:0] cnt;
    reg start_cnt;
    
    // Internal registers to avoid timing issues
    reg [7:0] dividend_abs_reg;
    reg [8:0] neg_divisor_reg;
    
    // Calculate absolute value in separate always block to ensure deterministic behavior
    always @(*) begin
        if (sign & dividend[7]) begin
            dividend_abs_reg = (~dividend) + 8'b00000001;
        end else begin
            dividend_abs_reg = dividend;
        end
    end
    
    // Calculate NEG_DIVISOR in separate always block
    always @(*) begin
        if (sign & divisor[7]) begin
            neg_divisor_reg = {1'b1, divisor};
        end else begin
            neg_divisor_reg = (~{1'b0, divisor}) + 9'b000000001;
        end
    end
    
    always @(posedge clk) begin
        if(rst) begin
            SR <= 16'b0;
            dividend_save <= 8'b0;
            divisor_save <= 8'b0;
            cnt <= 4'b0;
            start_cnt <= 1'b0;
            NEG_DIVISOR <= 9'b0;
        end
        else if(~start_cnt & opn_valid & ~res_valid) begin
            cnt <= 4'b1;
            start_cnt <= 1'b1;
        
            // Save inputs
            dividend_save <= dividend;
            divisor_save <= divisor;

            // Use pre-calculated values to avoid timing issues
            SR <= {7'b0000000, dividend_abs_reg, 1'b0}; 
            NEG_DIVISOR <= neg_divisor_reg;
        end
        else if(start_cnt) begin
            if(cnt[3]) begin    
                cnt <= 4'b0;
                start_cnt <= 1'b0;
                
                SR[15:8] <= mux_result[7:0];
                SR[0] <= CO;
            end
            else begin
                cnt <= cnt + 4'b1;
                SR <= {mux_result[6:0], SR[7:1], CO, 1'b0}; 
            end
        end
    end

    wire data_go;
    assign data_go = res_valid & res_ready;
    always @(posedge clk) begin
        res_valid <= rst     ? 1'b0 :
                     cnt[3]  ? 1'b1 :
                     data_go ? 1'b0 : res_valid;
    end
endmodule