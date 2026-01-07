import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载模型和tokenizer（全局加载一次，避免重复加载）
print("正在加载模型，请稍候...")
tokenizer = AutoTokenizer.from_pretrained("./Qwen/Qwen3-1.7B", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./Qwen/Qwen3-1.7B", device_map="auto", torch_dtype=torch.bfloat16)
#model = PeftModel.from_pretrained(model, model_id="./output/Qwen3-1.7B/checkpoint-1084")
print("模型加载完成！")

# 确定设备
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def predict_with_model(instruction, user_input, max_new_tokens=2048):
    """
    使用模型生成回复
    """
    # 构建消息
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_input}
    ]
    
    # 应用聊天模板
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    
    # 生成回复
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    
    # 解码回复
    generated_ids = [
        output_ids[len(input_ids):] 
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return response

def chat_interface(instruction, user_input, max_new_tokens):
    """
    Gradio界面处理函数
    """
    if not user_input.strip():
        return "请输入您的问题。"
    
    try:
        response = predict_with_model(instruction, user_input, int(max_new_tokens))
        return response
    except Exception as e:
        return f"生成回复时出错: {str(e)}"

def create_example(example_num):
    """
    创建示例
    """
    examples = [
        {
            "instruction": "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。",
            "input": "医生，我最近被诊断为糖尿病，听说碳水化合物的选择很重要，我应该选择什么样的碳水化合物呢？"
        },
        {
            "instruction": "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。",
            "input": "医生，我最近胃部不适，听说有几种抗溃疡药物可以治疗，您能详细介绍一下这些药物的分类、作用机制以及它们是如何影响胃黏膜的保护与损伤平衡的吗？"
        },
        {
            "instruction": "你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。",
            "input": "我最近被诊断出患有淋巴瘤，医生提到这可能导致发热。请问这是由于淋巴瘤组织的坏死和细胞破坏引起的吗？如果是，具体机制是什么？"
        }
    ]
    
    if example_num < len(examples):
        return examples[example_num]["instruction"], examples[example_num]["input"]
    return "", ""

# 创建Gradio界面
with gr.Blocks(title="Qwen3-1.7B医学助手", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🏥 Qwen3-1.7B医学助手-微调前
   基于Qwen3-1.7B模型微调的医学对话助手。您可以输入系统指令和您的问题，模型会生成详细的回答。
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 系统指令设置")
            instruction_input = gr.Textbox(
                label="系统指令",
                value="你是一个医学专家，你需要根据用户的问题，给出带有思考的回答。",
                lines=3,
                placeholder="请输入系统指令，定义助手的角色和回答风格..."
            )
            
            gr.Markdown("### 参数设置")
            max_tokens_slider = gr.Slider(
                minimum=100,
                maximum=4096,
                value=2048,
                step=100,
                label="最大生成长度"
            )
            
            gr.Markdown("### 示例")
            with gr.Row():
                example1_btn = gr.Button("示例1: 糖尿病饮食建议", variant="secondary", size="sm")
                example2_btn = gr.Button("示例2: 胃溃疡药物介绍", variant="secondary", size="sm")
                example3_btn = gr.Button("示例3: 淋巴瘤机制", variant="secondary", size="sm")
        
        with gr.Column(scale=2):
            gr.Markdown("### 对话界面")
            user_input = gr.Textbox(
                label="您的问题",
                lines=5,
                placeholder="请输入您的医学或健康相关问题..."
            )
            
            submit_btn = gr.Button("🚀 发送", variant="primary")
            clear_btn = gr.Button("🧹 清除", variant="secondary")
            
            output = gr.Textbox(
                label="助手回复",
                lines=10,
                interactive=False
            )
    
    # 示例按钮的事件处理
    example1_btn.click(
        fn=lambda: create_example(0),
        outputs=[instruction_input, user_input]
    )
    
    example2_btn.click(
        fn=lambda: create_example(1),
        outputs=[instruction_input, user_input]
    )
    
    example3_btn.click(
        fn=lambda: create_example(2),
        outputs=[instruction_input, user_input]
    )
    
    # 提交按钮的事件处理
    submit_btn.click(
        fn=chat_interface,
        inputs=[instruction_input, user_input, max_tokens_slider],
        outputs=output
    )
    
    # 清除按钮的事件处理
    clear_btn.click(
        fn=lambda: ("", "", ""),
        outputs=[instruction_input, user_input, output]
    )
    
    # 回车键提交
    user_input.submit(
        fn=chat_interface,
        inputs=[instruction_input, user_input, max_tokens_slider],
        outputs=output
    )
    
    gr.Markdown("""
    ### 使用说明
    1. 在"系统指令"中定义助手的角色
    2. 在"您的问题"中输入您的问题
    3. 点击"发送"按钮或按回车键获取回答
    4. 可以使用右侧的示例快速开始
    
    ### 注意事项
    - 本模型提供的信息仅供参考，不能替代专业医疗建议
    - 如有严重健康问题，请及时咨询专业医生
    - 模型回复可能存在延迟，请耐心等待
    """)

if __name__ == "__main__":
    # 启动Gradio界面
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,        # 端口号
        share=False,             # 是否创建公开链接
        debug=False              # 调试模式
    )